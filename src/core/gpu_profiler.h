#pragma once

#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <iomanip>

namespace spectra {

// Lightweight GPU profiler using CUDA events
class GpuProfiler {
public:
    GpuProfiler() = default;
    ~GpuProfiler() { shutdown(); }

    void init() {
        m_enabled = true;
    }

    void shutdown() {
        for (auto& [name, data] : m_timers) {
            if (data.startEvent) cudaEventDestroy(data.startEvent);
            if (data.endEvent) cudaEventDestroy(data.endEvent);
        }
        m_timers.clear();
    }

    void setEnabled(bool enabled) { m_enabled = enabled; }
    bool isEnabled() const { return m_enabled; }

    // Start timing a named section
    void begin(const std::string& name, cudaStream_t stream = 0) {
        if (!m_enabled) return;
        
        auto& data = getOrCreate(name);
        cudaEventRecord(data.startEvent, stream);
    }

    // End timing a named section
    void end(const std::string& name, cudaStream_t stream = 0) {
        if (!m_enabled) return;
        
        auto it = m_timers.find(name);
        if (it == m_timers.end()) return;
        
        auto& data = it->second;
        cudaEventRecord(data.endEvent, stream);
    }

    // Call once per frame after all GPU work is done (after sync)
    void endFrame() {
        if (!m_enabled) return;
        
        m_frameCount++;
        
        for (auto& [name, data] : m_timers) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, data.startEvent, data.endEvent);
            
            data.totalMs += ms;
            data.frameCount++;
            data.lastMs = ms;
            data.maxMs = std::max(data.maxMs, ms);
            data.minMs = std::min(data.minMs, ms);
        }
    }

    // Print statistics (call periodically, e.g., every 60 frames)
    void printStats() {
        if (!m_enabled || m_timers.empty()) return;
        
        std::cout << "\n=== GPU Profile (last " << m_frameCount << " frames) ===\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::left << std::setw(25) << "Section" 
                  << std::right << std::setw(10) << "Avg(ms)"
                  << std::setw(10) << "Last(ms)"
                  << std::setw(10) << "Min(ms)"
                  << std::setw(10) << "Max(ms)" << "\n";
        std::cout << std::string(65, '-') << "\n";
        
        float totalAvg = 0.0f;
        for (auto& [name, data] : m_timers) {
            float avg = data.frameCount > 0 ? data.totalMs / data.frameCount : 0.0f;
            totalAvg += avg;
            
            std::cout << std::left << std::setw(25) << name
                      << std::right << std::setw(10) << avg
                      << std::setw(10) << data.lastMs
                      << std::setw(10) << data.minMs
                      << std::setw(10) << data.maxMs << "\n";
        }
        std::cout << std::string(65, '-') << "\n";
        std::cout << std::left << std::setw(25) << "TOTAL"
                  << std::right << std::setw(10) << totalAvg << "\n\n";
    }

    // Reset all statistics
    void reset() {
        for (auto& [name, data] : m_timers) {
            data.totalMs = 0.0f;
            data.frameCount = 0;
            data.lastMs = 0.0f;
            data.maxMs = 0.0f;
            data.minMs = 999999.0f;
        }
        m_frameCount = 0;
    }

private:
    struct TimerData {
        cudaEvent_t startEvent = nullptr;
        cudaEvent_t endEvent = nullptr;
        float totalMs = 0.0f;
        float lastMs = 0.0f;
        float maxMs = 0.0f;
        float minMs = 999999.0f;
        uint32_t frameCount = 0;
    };

    TimerData& getOrCreate(const std::string& name) {
        auto it = m_timers.find(name);
        if (it != m_timers.end()) return it->second;
        
        TimerData data;
        cudaEventCreate(&data.startEvent);
        cudaEventCreate(&data.endEvent);
        m_timers[name] = data;
        return m_timers[name];
    }

    std::unordered_map<std::string, TimerData> m_timers;
    uint32_t m_frameCount = 0;
    bool m_enabled = false;
};

// RAII helper for scoped profiling
class GpuProfileScope {
public:
    GpuProfileScope(GpuProfiler& profiler, const std::string& name, cudaStream_t stream = 0)
        : m_profiler(profiler), m_name(name), m_stream(stream) {
        m_profiler.begin(m_name, m_stream);
    }
    ~GpuProfileScope() {
        m_profiler.end(m_name, m_stream);
    }
private:
    GpuProfiler& m_profiler;
    std::string m_name;
    cudaStream_t m_stream;
};

#define GPU_PROFILE_SCOPE(profiler, name, stream) \
    GpuProfileScope _gpu_scope_##__LINE__(profiler, name, stream)

} // namespace spectra
