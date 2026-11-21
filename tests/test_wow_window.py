"""The tracker keeps the last good handle so a client it can no longer classify is still found."""

from wvt import wow_window


def _patch(monkeypatch, user32, foreground, window_class, window_title, process_name, client_rect):
    monkeypatch.setattr(wow_window.winapi, "IS_WINDOWS", True)
    monkeypatch.setattr(wow_window.winapi, "user32", user32)
    monkeypatch.setattr(wow_window.winapi, "foreground_window", foreground)
    monkeypatch.setattr(wow_window.winapi, "window_class", window_class)
    monkeypatch.setattr(wow_window.winapi, "window_title", window_title)
    monkeypatch.setattr(wow_window.winapi, "process_name_of_window", process_name)
    monkeypatch.setattr(wow_window.winapi, "client_rect_on_screen", client_rect)


def test_last_good_handle_survives_cache_expiry(monkeypatch):
    wow_hwnd, app_hwnd = 111, 222
    foreground = {"hwnd": wow_hwnd}
    # while playing the window is recognisable; afterwards the app (focused) looks like nothing
    known = {"class": True}
    process = {"name": "wow.exe"}

    class User32:
        @staticmethod
        def IsWindow(hwnd):
            return hwnd == wow_hwnd

        @staticmethod
        def IsWindowVisible(hwnd):
            return False

        @staticmethod
        def EnumWindows(_cb, _lparam):
            return True  # no candidates outside the remembered handle

    _patch(
        monkeypatch, User32,
        foreground=lambda: foreground["hwnd"],
        window_class=lambda hwnd: "GxWindowClass" if known["class"] else "SomeOtherClass",
        window_title=lambda hwnd: "App",
        process_name=lambda hwnd: process["name"] if hwnd == wow_hwnd else "app.exe",
        client_rect=lambda hwnd: (0, 0, 100, 50) if hwnd == wow_hwnd else None,
    )

    tracker = wow_window.WowWindowTracker(["wow.exe"])
    found = tracker.find(max_age=0)
    assert found is not None and found.hwnd == wow_hwnd

    # the app gets focus: the foreground window is not WoW and the enum finds nothing
    known["class"] = False
    foreground["hwnd"] = app_hwnd
    again = tracker.find(max_age=0)
    assert again is not None and again.hwnd == wow_hwnd
    assert again.client == (0, 0, 100, 50)

    # a recycled handle now belongs to another process: not trusted, and never resurrected
    process["name"] = "notepad.exe"
    assert tracker.find(max_age=0) is None
    process["name"] = "wow.exe"
    assert tracker.find(max_age=0) is None
