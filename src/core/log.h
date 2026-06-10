#pragma once

//------------------------------------------------------------------------------
// Minimal runtime logging switch.
//
// Gates the recurring console output (per-frame diagnostics, path-guide build
// chatter) behind a runtime toggle so the console stays quiet by default.
// One-shot initialization/load messages and warnings stay unconditional.
// Toggled with F6 (see Application::keyCallback).
//------------------------------------------------------------------------------

#include <atomic>

namespace spectra {

inline std::atomic<bool> g_verboseLogging{false};

inline bool verboseLogging() {
    return g_verboseLogging.load(std::memory_order_relaxed);
}

} // namespace spectra
