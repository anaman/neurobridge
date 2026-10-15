"""Visual presentation for stimulus-locked tests.

Timing model
------------
A visual stimulus is "on" the instant the GPU/compositor accepts the frame,
which in pygame is observable as the moment ``pygame.display.flip()`` returns
(a double-buffered flip, so the new frame is queued for display before we
timestamp).  Every ``show_*`` therefore returns ``time.perf_counter()`` taken
*immediately after* the flip, and the runner logs that number as the stimulus
onset -- not the time we started drawing, which would be early by the render
cost.

Keeping acquisition alive
-------------------------
The TH21A delivers 10 samples per notification at ~25 notifications/s.  A
``time.sleep(0.5)`` in the middle of a trial would let the BLE receive queue
back up and, worse, would delay every subsequent timestamp.  :meth:`pump`
therefore never sleeps more than ~5 ms at a time: each slice drains SDL events
and runs ``poll_cb`` (the runner passes its device poller), so EEG acquisition
and the window event loop both stay live during a 30 s break.

Fullscreen and the desktop
--------------------------
Fullscreen is the default whenever a display is available: the stimuli must be
the only thing on the screen, with no panels, wallpaper or mouse cursor moving
behind them.  Three details make that actually true:

* the window is sized from the **real desktop size** (``get_desktop_sizes``),
  not a hard-coded 1280x800, and created with ``FULLSCREEN | SCALED``;
* ``VIDEOEXPOSE`` / ``WINDOWFOCUSLOST`` re-assert fullscreen, because some
  window managers quietly demote a fullscreen window;
* the screensaver / display blanking is inhibited for the session through the
  first mechanism that works (D-Bus ScreenSaver, ``xdg-screensaver``,
  ``systemd-inhibit``) and always released on close.

Headless operation
------------------
On a machine with no ``DISPLAY`` (CI, a server, a broken X session) pygame
cannot open a window.  We never raise: the presenter falls back to a
``terminal`` backend that prints each stimulus and sleeps, and reports
``active == False`` so the runner knows there is no display and can
auto-advance without waiting for a keypress.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..device.base import clock

#: Fonts that exist on a stock Linux Mint install, best first.  DejaVu Sans is
#: pulled in by nearly every desktop; the bitmap default is the last resort.
_FONT_CHAIN: tuple[str, ...] = (
    "dejavusans", "dejavu sans", "liberationsans", "liberation sans",
    "freesans", "noto sans", "ubuntu", "arial", "helvetica",
)

#: Environment overrides tried in order when opening the probe window.  Some
#: X servers cannot create a GLX context but can create an EGL one (the Mint
#: box this was developed on is exactly that case), so FORCE_EGL is the second
#: attempt rather than a blanket setting.
_PROBE_ENVS: tuple[dict[str, str], ...] = (
    {},
    {"SDL_VIDEO_X11_FORCE_EGL": "1"},
)

#: Fullscreen re-assert policy.  Bounded on purpose: a compositor that keeps
#: demoting the window must not be able to stall stimulus timing.
_REASSERT_MIN_INTERVAL_S = 1.0
_REASSERT_MAX_ATTEMPTS = 3
_REASSERT_SLOW_S = 2.0

#: Process-wide pygame session.  Re-running ``pygame.display.init()`` /
#: ``pygame.quit()`` once per presenter segfaults SDL on this host under
#: fullscreen EGL (the second init aborts with an X GLX BadValue), so SDL is
#: initialised exactly once and never torn down; each presenter simply calls
#: ``set_mode`` again, which is safe.  This is what lets a process run several
#: sessions in a row (GUI, test harness) without crashing.
_PYGAME: Any = None
_PYGAME_LOCK = threading.Lock()
_PROBE_CACHE: dict[tuple[int, int, bool], tuple[bool, dict[str, str]]] = {}


def _pygame_session() -> Any:
    """Initialise pygame display+font once per process; return the module."""
    global _PYGAME
    with _PYGAME_LOCK:
        if _PYGAME is None:
            import pygame
            pygame.display.init()
            pygame.font.init()
            _PYGAME = pygame
        return _PYGAME


def _probe_display(size: tuple[int, int], fullscreen: bool) -> tuple[bool, dict[str, str]]:
    """Try to open a hidden window in a child process.

    SDL can abort the whole process on a broken X server (a C-level X error is
    not catchable in Python), so the only safe way to know whether a window is
    possible is to attempt it somewhere disposable.  Returns ``(ok, env)`` where
    ``env`` are the environment overrides the parent must apply before its own
    ``set_mode`` call.
    """
    cache_key = (int(size[0]), int(size[1]), bool(fullscreen))
    if cache_key in _PROBE_CACHE:
        return _PROBE_CACHE[cache_key]
    code = (
        "import os\n"
        "os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')\n"
        "import pygame\n"
        "pygame.display.init(); pygame.font.init()\n"
        "flags = (pygame.FULLSCREEN | pygame.SCALED if %r else 0) | pygame.HIDDEN\n"
        f"pygame.display.set_mode(({int(size[0])},{int(size[1])}), flags)\n"
        "print('PROBE_OK')\n"
        "pygame.quit()\n"
    ) % bool(fullscreen)
    for overrides in _PROBE_ENVS:
        env = dict(os.environ)
        env.update(overrides)
        try:
            proc = subprocess.run([sys.executable, "-c", code], env=env,
                                  capture_output=True, timeout=20.0)
        except Exception:
            continue
        if proc.returncode == 0 and b"PROBE_OK" in (proc.stdout or b""):
            _PROBE_CACHE[cache_key] = (True, overrides)
            return True, overrides
    _PROBE_CACHE[cache_key] = (False, {})
    return False, {}


class VisualPresenter:
    """Window/terminal stimulus display with post-flip onset timestamps."""

    def __init__(self, size: tuple[int, int] = (1280, 800), fullscreen: bool = False,
                 backend: str = "auto", bg: tuple[int, int, int] = (18, 18, 20),
                 echo: Callable[[str], None] | None = None) -> None:
        #: Windowed size (settings.screen_size).  Fullscreen ignores it and uses
        #: the real desktop size instead; it is kept as the windowed fallback.
        self._requested_size = (int(size[0]), int(size[1]))
        self.size = (int(size[0]), int(size[1]))
        self.bg = tuple(int(c) for c in bg)
        self.fullscreen = bool(fullscreen)
        self._echo = echo
        self._sdl_env: dict[str, str] = {}
        self._sdl_driver = ""
        self._open_path = "none"     # probe env / in-process mode actually used
        self._pg: Any = None
        self._screen: Any = None
        self._fonts: dict[tuple[int, bool], Any] = {}
        self._images: "OrderedDict[tuple[str, float, tuple[int, int]], Any]" = OrderedDict()
        self._image_max = 32
        self._active = False
        self._backend = "terminal"
        self._warnings: list[str] = []
        self._closed = False
        self._desktop_size: tuple[int, int] | None = None
        self._cursor_hidden = False
        self._last_reassert = 0.0
        self._need_reassert = False
        self._reassert_attempts = 0
        self._reassert_disabled = False
        # screensaver inhibition state
        self._ss_method = "none"
        self._ss_cookie: str | None = None
        self._ss_backend: str = ""
        self._ss_proc: subprocess.Popen | None = None
        self._resolve(backend)

    # ------------------------------------------------------------ resolution
    def _resolve(self, backend: str) -> None:
        want = (backend or "auto").strip().lower()
        if want in ("terminal", "print", "none", "off", "headless"):
            self._backend = "terminal"
            return
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
                           or os.environ.get("SDL_VIDEODRIVER"))
        if not has_display:
            self._warn("no DISPLAY/WAYLAND_DISPLAY/SDL_VIDEODRIVER; using terminal backend")
            self._backend = "terminal"
            return
        # A broken X server can abort the *process* inside SDL (BadValue from
        # X_GLXCreateContext is a C-level X error, not a Python exception), so a
        # window is probed in a short-lived child process first.
        ok, overrides = _probe_display(self.size, self.fullscreen)
        if not ok:
            self._warn("pygame could not open a display (probe failed); using terminal")
            self._backend = "terminal"
            return
        # Apply the working SDL env in the PARENT too, not just in the probe:
        # on this host plain X11/GLX hard-aborts the process, so a real session
        # must take the same path the probe proved works.
        os.environ.update(overrides)
        self._sdl_env = dict(overrides)
        try:
            pygame = _pygame_session()
        except Exception as exc:
            self._warn(f"pygame unavailable ({exc.__class__.__name__}: {exc}); using terminal")
            self._backend = "terminal"
            return
        try:
            if self.fullscreen:
                self._desktop_size = self._desktop_size_of(pygame)
                if self._desktop_size:
                    self.size = self._desktop_size
            self._screen = self._open_window(pygame)
            pygame.display.set_caption("NeuroBridge")
            try:
                pygame.mouse.set_visible(False)
                self._cursor_hidden = True
            except Exception:
                pass
            self._screen.fill(self.bg)
            pygame.display.flip()
            pygame.event.pump()
        except Exception as exc:
            self._warn(f"cannot open a display ({exc.__class__.__name__}: {exc}); using terminal")
            self._screen = None
            self._backend = "terminal"
            return
        self._pg = pygame
        self.size = tuple(self._screen.get_size())
        self._active = True
        self._backend = "pygame"
        try:
            self._sdl_driver = str(pygame.display.get_driver())
        except Exception:
            self._sdl_driver = ""
        self._open_path = ("probe-env=" + ",".join(f"{k}={v}" for k, v in self._sdl_env.items())
                           if self._sdl_env else "probe-env=default") + f" driver={self._sdl_driver}"
        if self._desktop_size is None:
            self._desktop_size = self._desktop_size_of(pygame)

    def _open_window(self, pygame: Any) -> Any:
        """Create the window, preferring true fullscreen, then a plain window.

        Fullscreen uses the *desktop* size (a 1280x800 ``FULLSCREEN`` window is
        a 1280x800 window in the corner on a 1920x1080 panel); the windowed
        fallback uses the configured window size instead, so a failed
        fullscreen does not silently produce a desktop-sized window.
        """
        if self.fullscreen:
            for flags in (pygame.FULLSCREEN | getattr(pygame, "SCALED", 0),
                          pygame.FULLSCREEN):
                try:
                    return pygame.display.set_mode(self.size, flags)
                except Exception as exc:
                    self._warn(f"fullscreen mode failed ({exc.__class__.__name__}); retrying")
        return pygame.display.set_mode(self._requested_size, 0)

    @staticmethod
    def _desktop_size_of(pygame: Any) -> tuple[int, int] | None:
        """Real desktop resolution: fullscreen must cover the actual panel."""
        try:
            sizes = pygame.display.get_desktop_sizes()
            if sizes:
                w, h = sizes[0]
                if int(w) > 0 and int(h) > 0:
                    return (int(w), int(h))
        except Exception:
            pass
        try:
            info = pygame.display.Info()
            if info.current_w > 0 and info.current_h > 0:
                return (int(info.current_w), int(info.current_h))
        except Exception:
            pass
        return None

    def _assert_fullscreen(self) -> None:
        """Re-apply fullscreen after a focus loss, *outside* the event loop.

        Creating a window from inside ``poll_events`` deadlocks against the
        compositor on this host (measured stalls of 1.3 s, 3.9 s, 11.7 s as each
        re-creation generated another focus event).  So the event loop only sets
        a flag, and the re-assert runs at most a few times, rate-limited, just
        before a frame is flipped -- with no extra fill/flip of its own, so the
        current stimulus frame is never blanked.
        """
        if (not self._active or not self.fullscreen or self._pg is None
                or self._reassert_disabled):
            return
        now = clock()
        if now - self._last_reassert < _REASSERT_MIN_INTERVAL_S:
            self._need_reassert = True       # retry on a later frame
            return
        if self._reassert_attempts >= _REASSERT_MAX_ATTEMPTS:
            self._reassert_disabled = True
            return
        self._last_reassert = now
        self._reassert_attempts += 1
        try:
            t0 = clock()
            screen = self._open_window(self._pg)
            if screen is not None:
                self._screen = screen
                self.size = tuple(screen.get_size())
            if clock() - t0 > _REASSERT_SLOW_S:
                # The compositor fought us; stop trying rather than stall
                # stimulus timing with more window re-creations.
                self._reassert_disabled = True
        except Exception:
            self._reassert_disabled = True

    def _warn(self, msg: str) -> None:
        if msg not in self._warnings:
            self._warnings.append(msg)

    def _echo_print(self, text: str) -> None:
        if self._echo is not None:
            try:
                self._echo(text)
                return
            except Exception:
                pass
        print(text)

    # ------------------------------------------------------------ properties
    @property
    def active(self) -> bool:
        """True only when a real window is being drawn (pygame backend)."""
        return bool(self._active)

    @property
    def backend(self) -> str:
        return self._backend if not self._closed else "closed"

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    @property
    def desktop_size(self) -> tuple[int, int] | None:
        return self._desktop_size

    @property
    def requested_size(self) -> tuple[int, int]:
        return tuple(self._requested_size)

    @property
    def sdl_driver(self) -> str:
        return self._sdl_driver

    @property
    def sdl_env(self) -> dict[str, str]:
        return dict(self._sdl_env)

    @property
    def open_path(self) -> str:
        """Which SDL path was taken (probe env + driver), for session meta."""
        return self._open_path

    @property
    def screen_size(self) -> tuple[int, int]:
        return tuple(self.size)

    @property
    def cursor_hidden(self) -> bool:
        return bool(self._cursor_hidden)

    @property
    def screensaver_method(self) -> str:
        return self._ss_method

    @property
    def fonts(self) -> dict[tuple[int, bool], Any]:
        return self._fonts

    # ------------------------------------------------------------ screensaver
    def inhibit_screensaver(self) -> str:
        """Stop the screensaver / display blanking from interrupting a session.

        Tried in order, first success wins: D-Bus ``ScreenSaver.Inhibit`` (via
        ``gdbus`` then ``dbus-send``), ``xdg-screensaver suspend``, then a
        ``systemd-inhibit sleep`` child.  Failure is never fatal -- a session
        must not depend on a desktop service -- and the method (or ``none``) is
        reported so the operator can see whether blanking was blocked.
        """
        if self._ss_method != "none" or not self._active:
            return self._ss_method
        for attempt in (self._ss_dbus, self._ss_xdg, self._ss_systemd):
            try:
                method = attempt()
            except Exception:
                method = ""
            if method:
                self._ss_method = method
                return method
        self._ss_method = "none"
        return "none"

    def _ss_dbus(self) -> str:
        service = "org.freedesktop.ScreenSaver"
        path = "/org/freedesktop/ScreenSaver"
        method = "org.freedesktop.ScreenSaver.Inhibit"
        gdbus = shutil.which("gdbus")
        if gdbus:
            out = self._run_capture([gdbus, "call", "--session", "--dest", service,
                                     "--object-path", path, "--method", method,
                                     "NeuroBridge", "EEG session"])
            cookie = _parse_uint32(out)
            if cookie is not None:
                self._ss_cookie = cookie
                self._ss_backend = "gdbus"
                return f"{service} (gdbus)"
        dbus_send = shutil.which("dbus-send")
        if dbus_send:
            out = self._run_capture([dbus_send, "--session", "--print-reply",
                                     f"--dest={service}", path, method,
                                     "string:NeuroBridge", "string:EEG session"])
            cookie = _parse_uint32(out)
            if cookie is not None:
                self._ss_cookie = cookie
                self._ss_backend = "dbus-send"
                return f"{service} (dbus-send)"
        if gdbus:                            # GNOME fallback: flags 8 == idle
            out = self._run_capture([gdbus, "call", "--session", "--dest",
                                     "org.gnome.SessionManager", "--object-path",
                                     "/org/gnome/SessionManager",
                                     "--method", "org.gnome.SessionManager.Inhibit",
                                     "NeuroBridge", "0", "8", "EEG session"])
            cookie = _parse_uint32(out)
            if cookie is not None:
                self._ss_cookie = cookie
                self._ss_backend = "gnome-gdbus"
                return "org.gnome.SessionManager (gdbus)"
        return ""

    def _ss_xdg(self) -> str:
        exe = shutil.which("xdg-screensaver")
        xid = self._window_id()
        if not exe or not xid:
            return ""
        out = self._run_capture([exe, "suspend", str(xid)])
        if out is not None:                  # xdg-screensaver exits 0 on success
            self._ss_backend = "xdg"
            return f"xdg-screensaver suspend {xid}"
        return ""

    def _ss_systemd(self) -> str:
        exe = shutil.which("systemd-inhibit")
        if not exe:
            return ""
        try:
            self._ss_proc = subprocess.Popen(
                [exe, "--what=idle:sleep", "--mode=block",
                 "--why=NeuroBridge EEG session", "sleep", "86400"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._ss_backend = "systemd-inhibit"
            return "systemd-inhibit (idle:sleep)"
        except Exception:
            return ""

    def release_screensaver(self) -> None:
        """Release whatever :meth:`inhibit_screensaver` acquired (idempotent)."""
        backend, cookie = self._ss_backend, self._ss_cookie
        proc = self._ss_proc
        self._ss_proc = None
        self._ss_cookie = None
        if backend in ("gdbus", "dbus-send", "gnome-gdbus") and cookie is not None:
            gdbus = shutil.which("gdbus")
            dbus_send = shutil.which("dbus-send")
            if backend == "gnome-gdbus" and gdbus:
                self._run_capture([gdbus, "call", "--session", "--dest",
                                   "org.gnome.SessionManager", "--object-path",
                                   "/org/gnome/SessionManager", "--method",
                                   "org.gnome.SessionManager.Uninhibit", str(cookie)])
            elif gdbus:
                self._run_capture([gdbus, "call", "--session", "--dest",
                                   "org.freedesktop.ScreenSaver", "--object-path",
                                   "/org/freedesktop/ScreenSaver", "--method",
                                   "org.freedesktop.ScreenSaver.UnInhibit", str(cookie)])
            elif dbus_send:
                self._run_capture([dbus_send, "--session", "--print-reply",
                                   "--dest=org.freedesktop.ScreenSaver",
                                   "/org/freedesktop/ScreenSaver",
                                   "org.freedesktop.ScreenSaver.UnInhibit",
                                   f"uint32:{cookie}"])
        elif backend == "xdg":
            exe = shutil.which("xdg-screensaver")
            xid = self._window_id()
            if exe and xid:
                self._run_capture([exe, "resume", str(xid)])
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._ss_backend = ""

    def _window_id(self) -> str | None:
        if self._pg is None or self._screen is None:
            return None
        try:
            info = self._pg.display.get_wm_info()
            xid = info.get("window") if isinstance(info, dict) else None
            return str(int(xid)) if xid else None
        except Exception:
            return None

    @staticmethod
    def _run_capture(cmd: list[str]) -> str | None:
        """Run a short command; return its output, or None if it failed."""
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        return (proc.stdout or "") + (proc.stderr or "")

    # ------------------------------------------------------------ fonts/images
    def _font(self, size: int, bold: bool = False) -> Any:
        key = (int(size), bool(bold))
        hit = self._fonts.get(key)
        if hit is not None:
            return hit
        pg = self._pg
        font = None
        if pg is not None:
            for names in (list(_FONT_CHAIN), [_FONT_CHAIN[0]]):
                try:
                    font = pg.font.SysFont(names, int(size), bold=bold)
                except Exception:
                    font = None
                if font is not None and getattr(font, "get_height", lambda: 0)() > 0:
                    break
            if font is None:
                try:
                    font = pg.font.Font(None, int(size))
                except Exception:
                    font = None
        self._fonts[key] = font
        return font

    def _load_image(self, path: str | Path, max_frac: float) -> Any:
        pg = self._pg
        key = (str(path), float(max_frac), tuple(self.size))
        hit = self._images.get(key)
        if hit is not None:
            self._images.move_to_end(key)
            return hit
        try:
            raw = pg.image.load(str(path))
            try:
                raw = raw.convert()
            except Exception:
                pass
            iw, ih = raw.get_size()
            sw, sh = self.size
            if iw > 0 and ih > 0:
                scale = min((sw * max_frac) / iw, (sh * max_frac) / ih)
                if abs(scale - 1.0) > 1e-3:
                    nw = max(1, int(round(iw * scale)))
                    nh = max(1, int(round(ih * scale)))
                    raw = pg.transform.smoothscale(raw, (nw, nh))
        except Exception as exc:
            self._warn(f"cannot load image {Path(path).name} ({exc.__class__.__name__})")
            raw = None
        self._images[key] = raw
        while len(self._images) > self._image_max:
            self._images.popitem(last=False)
        return raw

    # ------------------------------------------------------------ drawing
    def _begin_frame(self) -> None:
        """Run any deferred fullscreen re-assert *before* the frame is drawn.

        Re-creating the window discards the surface contents, so doing it here
        (never between drawing and flipping) means the stimulus is always
        painted onto the window that is actually shown.
        """
        if self._need_reassert:
            self._need_reassert = False
            self._assert_fullscreen()

    def _present_surface(self) -> float:
        """Flip and return the post-flip monotonic timestamp."""
        self._pg.display.flip()
        return clock()

    def show_text(self, text: str, *, font_size: int = 30,
                  duration_s: float | None = None,
                  color: tuple[int, int, int] = (235, 235, 235)) -> float:
        if not self._active:
            return self._terminal_text(text, duration_s)
        self._begin_frame()
        screen = self._screen
        screen.fill(self.bg)
        font = self._font(font_size)
        lines = self._wrap(str(text), font, int(self.size[0] * 0.86))
        line_h = max(1, (font.get_height() if font is not None else font_size + 4) + 6)
        total = line_h * max(1, len(lines))
        y = (self.size[1] - total) // 2
        for line in lines:
            surf = font.render(line, True, tuple(color)) if font is not None else None
            if surf is not None:
                screen.blit(surf, ((self.size[0] - surf.get_width()) // 2, y))
            y += line_h
        t = self._present_surface()
        if duration_s:
            self.pump(float(duration_s))
        return t

    def show_image(self, path: str | Path, *, duration_s: float | None = None,
                   max_frac: float = 0.8) -> float:
        if not self._active:
            return self._terminal_text(f"[image] {path}", duration_s)
        self._begin_frame()
        surf = self._load_image(path, float(max_frac))
        self._screen.fill(self.bg)
        if surf is not None:
            w, h = surf.get_size()
            self._screen.blit(surf, ((self.size[0] - w) // 2, (self.size[1] - h) // 2))
        else:
            self._draw_centered_text(f"[missing image] {Path(path).name}", 24)
        t = self._present_surface()
        if duration_s:
            self.pump(float(duration_s))
        return t

    def show_fixation(self, char: str = "+", duration_s: float | None = None) -> float:
        if not self._active:
            return self._terminal_text(str(char), duration_s)
        self._begin_frame()
        self._screen.fill(self.bg)
        self._draw_centered_text(str(char), int(self.size[1] * 0.09))
        t = self._present_surface()
        if duration_s:
            self.pump(float(duration_s))
        return t

    def show_blank(self, duration_s: float | None = None) -> float:
        if not self._active:
            return self._terminal_text("", duration_s)
        self._begin_frame()
        self._screen.fill(self.bg)
        t = self._present_surface()
        if duration_s:
            self.pump(float(duration_s))
        return t

    def flash(self, color: tuple[int, int, int] = (255, 255, 255),
              duration_s: float = 0.05) -> float:
        """Full-field flash: the visual-latency calibration stimulus (T005)."""
        if not self._active:
            return self._terminal_text(f"[flash {tuple(color)}]", duration_s)
        self._begin_frame()
        self._screen.fill(tuple(int(c) for c in color))
        t = self._present_surface()
        if duration_s:
            self.pump(float(duration_s))
        self._screen.fill(self.bg)
        self._present_surface()
        return t

    def show_countdown(self, seconds: float, label: str = "") -> None:
        """Countdown used before instructions/breaks in timer-driven mode."""
        seconds = max(0.0, float(seconds))
        if seconds <= 0:
            return
        steps = max(1, int(round(seconds)))
        per = seconds / steps
        for k in range(steps, 0, -1):
            txt = f"{label}\n{k}" if label else str(k)
            if self._active:
                self.show_text(txt, font_size=44)
            else:
                self._echo_print(f"[countdown] {label} {k}" if label else f"[countdown] {k}")
            self.pump(per)

    def show_progress(self, frac: float, label: str = "") -> float:
        frac = min(1.0, max(0.0, float(frac)))
        if not self._active:
            self._pg_progress_print(frac, label)
            return clock()
        self._begin_frame()
        self._screen.fill(self.bg)
        if label:
            self._draw_centered_text(label, 26, dy=-40)
        pg = self._pg
        sw, sh = self.size
        m = int(sw * 0.1)
        bar = pg.Rect(m, sh - int(sh * 0.14), sw - 2 * m, 18)
        pg.draw.rect(self._screen, (60, 60, 66), bar, border_radius=6)
        inner = pg.Rect(bar.x, bar.y, int(bar.w * frac), bar.h)
        pg.draw.rect(self._screen, (110, 170, 240), inner, border_radius=6)
        return self._present_surface()

    def flicker(self, hz: float, seconds: float, *,
                poll_cb: Callable[[], Any] | None = None) -> float:
        """Full-field square-wave flicker at ``hz`` for ``seconds`` (T103).

        Entrainment needs a real-time flicker, so this is one of the few places
        that runs at wall-clock speed: each half-cycle flips the frame and then
        polls the acquisition callback every ~2 ms so a 10 s train never stalls
        the BLE stream.  Returns the timestamp of the first flip.
        """
        seconds = max(0.0, float(seconds))
        hz = max(0.1, float(hz))
        if not self._active:
            self._echo_print(f"[flicker] {hz:g} Hz for {seconds:.2f}s")
            self.pump(seconds, poll_cb=poll_cb)
            return clock()
        half = 0.5 / hz
        end = clock() + seconds
        first: float | None = None
        on = True
        while clock() < end:
            self._screen.fill((252, 252, 252) if on else self.bg)
            t = self._present_surface()
            if first is None:
                first = t
            on = not on
            target = clock() + half
            while clock() < target:
                if poll_cb is not None:
                    try:
                        poll_cb()
                    except Exception:
                        pass
                else:
                    self.poll_events()
                if clock() >= end:
                    break
                time.sleep(min(0.002, max(0.0, target - clock())))
        self._screen.fill(self.bg)
        self._present_surface()
        return first if first is not None else clock()

    def _draw_centered_text(self, text: str, font_size: int,
                            color: tuple[int, int, int] = (235, 235, 235),
                            dy: int = 0) -> None:
        font = self._font(font_size)
        if font is None:
            return
        surf = font.render(text, True, tuple(color))
        self._screen.blit(surf, ((self.size[0] - surf.get_width()) // 2,
                                 (self.size[1] - surf.get_height()) // 2 + int(dy)))

    @staticmethod
    def _wrap(text: str, font: Any, max_px: int) -> list[str]:
        out: list[str] = []
        for para in str(text).splitlines() or [""]:
            if not para.strip():
                out.append("")
                continue
            words = para.split()
            line = ""
            for w in words:
                trial = f"{line} {w}".strip()
                width = font.size(trial)[0] if font is not None else len(trial) * 9
                if line and width > max_px:
                    out.append(line)
                    line = w
                else:
                    line = trial
            out.append(line)
        return out

    # ------------------------------------------------------------ events
    def poll_events(self) -> list[dict[str, Any]]:
        """Return pending key/quit events, each stamped with ``t`` (monotonic).

        The timestamp is taken when the event is *read*, not when SDL queued it:
        pygame gives no kernel timestamp, and reading happens within ~5 ms of the
        key because ``pump`` slices are that short.  That is the reaction-time
        resolution this system can honestly claim.

        ``WINDOWFOCUSLOST`` is handled here (not returned): it means the window
        manager demoted us, so a fullscreen re-assert is *scheduled* for the next
        frame.  The re-assert is deliberately not performed inside this loop --
        re-creating a window from the event loop deadlocks the compositor.
        """
        if not self._active:
            return []
        pg = self._pg
        out: list[dict[str, Any]] = []
        try:
            pg.event.pump()
            events = pg.event.get()
        except Exception:
            return out
        focus_lost = getattr(pg, "WINDOWFOCUSLOST", -2)
        for ev in events:
            if ev.type == focus_lost:
                self._need_reassert = True
            if ev.type == pg.QUIT:
                out.append({"type": "quit", "t": clock()})
            elif ev.type == pg.KEYDOWN:
                try:
                    key = pg.key.name(ev.key)
                except Exception:
                    key = str(ev.key)
                out.append({"type": "key", "key": key, "t": clock(),
                            "mods": int(getattr(ev, "mod", 0)),
                            "unicode": getattr(ev, "unicode", "")})
        return out

    def pump(self, seconds: float, *, poll_cb: Callable[[], Any] | None = None) -> list[dict[str, Any]]:
        """Display for ``seconds`` while calling ``poll_cb`` every <= 5 ms slice.

        Returning the collected key/quit events lets the runner react to a
        response *and* to an abort mid-pump.  A slice is deliberately short:
        long sleeps are what make BLE acquisition back up and windows freeze.
        """
        events: list[dict[str, Any]] = []
        end = clock() + max(0.0, float(seconds))
        while True:
            events.extend(self.poll_events())
            if poll_cb is not None:
                try:
                    poll_cb()
                except Exception:
                    pass
            rem = end - clock()
            if rem <= 0:
                break
            time.sleep(min(0.005, rem))
        return events

    # ------------------------------------------------------------ teardown
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._active = False
        try:
            self.release_screensaver()
        except Exception:
            pass
        if self._pg is not None:
            try:
                self._pg.mouse.set_visible(True)     # never leave a hidden cursor
                self._cursor_hidden = False
            except Exception:
                pass
            # Deliberately no pygame.display.quit()/pygame.quit(): the SDL
            # session is process-wide, and tearing it down makes the *next*
            # fullscreen window abort inside SDL on this host.
        self._screen = None
        self._images.clear()
        self._fonts.clear()

    # ------------------------------------------------------------ terminal
    def _terminal_text(self, text: str, duration_s: float | None) -> float:
        body = str(text).replace("\n", " | ")
        self._echo_print(f"[stimulus] {body}" if body else "[stimulus] (blank)")
        t = clock()
        if duration_s:
            self.pump(float(duration_s))
        return t

    def _pg_progress_print(self, frac: float, label: str) -> None:
        n = 30
        filled = int(round(n * frac))
        self._echo_print(f"[progress] {label} [{'#' * filled}{'.' * (n - filled)}] "
                         f"{frac * 100:5.1f}%")

    # ------------------------------------------------------------ helpers
    def draw(self, items: Iterable[Sequence[Any]]) -> float:
        """Escape hatch for custom frames: list of (text, font_size, color, y)."""
        if not self._active:
            return self._terminal_text(" | ".join(str(i[0]) for i in items), None)
        self._screen.fill(self.bg)
        for item in items:
            text, size = str(item[0]), int(item[1] if len(item) > 1 else 30)
            color = tuple(item[2]) if len(item) > 2 and item[2] else (235, 235, 235)
            y = item[3] if len(item) > 3 else None
            font = self._font(size)
            if font is None:
                continue
            surf = font.render(text, True, color)
            yy = int(y) if y is not None else (self.size[1] - surf.get_height()) // 2
            self._screen.blit(surf, ((self.size[0] - surf.get_width()) // 2, yy))
        return self._present_surface()


def _parse_uint32(text: str | None) -> str | None:
    """Pull a D-Bus ``uint32`` cookie out of gdbus/dbus-bus output."""
    if not text:
        return None
    import re
    m = re.search(r"uint32\s+(\d+)", text)
    if m:
        return m.group(1)
    m = re.search(r"\((\d+),?\)", text)
    return m.group(1) if m else None
