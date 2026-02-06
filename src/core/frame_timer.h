#pragma once

#include <chrono>
#include <cstdint>

namespace spectra {

struct FrameTimer {
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;

    TimePoint frameStart;
    TimePoint lastFrame;
    double frameTimeMs = 0.0;
    double deltaTime = 0.0;
    double fps = 0.0;
    uint64_t frameCount = 0;

    static constexpr int SAMPLE_COUNT = 60;
    double samples[SAMPLE_COUNT] = {};
    int sampleIndex = 0;

    void beginFrame() {
        frameStart = Clock::now();
        if (frameCount > 0) {
            deltaTime = std::chrono::duration<double>(frameStart - lastFrame).count();
        } else {
            deltaTime = 1.0 / 60.0;
        }
        lastFrame = frameStart;
    }

    void endFrame() {
        auto now = Clock::now();
        frameTimeMs = std::chrono::duration<double, std::milli>(now - frameStart).count();

        samples[sampleIndex] = frameTimeMs;
        sampleIndex = (sampleIndex + 1) % SAMPLE_COUNT;

        double sum = 0.0;
        for (int i = 0; i < SAMPLE_COUNT; i++) {
            sum += samples[i];
        }
        double avgMs = sum / SAMPLE_COUNT;
        fps = 1000.0 / avgMs;

        frameCount++;
    }
};

} // namespace spectra
