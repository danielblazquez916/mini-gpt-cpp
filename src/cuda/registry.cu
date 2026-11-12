#include "mini_gpt/cuda/registry.cuh"

#include "mini_gpt/cuda/memory.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace mini_gpt::cuda {
namespace {

struct Entry {
    void* device = nullptr;
    std::size_t bytes = 0;
    bool host_dirty = false;
    bool host_mutable = false;
    bool device_dirty = false;
};

std::unordered_map<const void*, Entry> entries;
std::unordered_map<void*, void*> aliases;
std::uint64_t uploaded_bytes = 0;
std::uint64_t downloaded_bytes = 0;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

void register_host(const void* host, std::size_t bytes, bool host_dirty) {
    if (host == nullptr) {
        return;
    }
    if (const auto it = entries.find(host); it != entries.end()) {
        deallocate_device(it->second.device, it->second.bytes);
        entries.erase(it);
    }
    void* device = allocate_device(bytes);
    if (!host_dirty) {
        check_cuda(cudaMemset(device, 0, bytes), "initialize zero tensor");
    }
    entries.emplace(host, Entry{device, bytes, host_dirty, false});
}

void unregister_host(const void* host) {
    const auto it = entries.find(host);
    if (it == entries.end()) {
        return;
    }
    deallocate_device(it->second.device, it->second.bytes);
    entries.erase(it);
}

void move_host_registration(const void* old_host, const void* new_host) {
    if (old_host == new_host) {
        return;
    }
    const auto it = entries.find(old_host);
    if (it == entries.end()) {
        return;
    }
    Entry entry = std::move(it->second);
    entries.erase(it);
    if (const auto existing = entries.find(new_host); existing != entries.end()) {
        deallocate_device(existing->second.device, existing->second.bytes);
        entries.erase(existing);
    }
    entries.emplace(new_host, std::move(entry));
}

bool is_registered(const void* host) {
    return entries.find(host) != entries.end();
}


void* device_for_host(const void* host) {
    const auto it = entries.find(host);
    if (it == entries.end()) {
        return nullptr;
    }
    Entry& entry = it->second;
    if (entry.host_dirty || entry.host_mutable) {
        check_cuda(
            cudaMemcpy(entry.device, host, entry.bytes, cudaMemcpyHostToDevice),
            "synchronize host to device");
        uploaded_bytes += entry.bytes;
        entry.host_dirty = false;
        entry.device_dirty = false;
    }
    return entry.device;
}

void* alias_for_buffer(void* buffer) {
    const auto it = aliases.find(buffer);
    return it == aliases.end() ? buffer : it->second;
}

void set_alias(void* buffer, void* device) {
    aliases[buffer] = device;
}

void clear_alias(void* buffer) {
    aliases.erase(buffer);
}

void sync_host(const void* host) {
    const auto it = entries.find(host);
    if (it == entries.end()) {
        return;
    }
    Entry& entry = it->second;
    if (entry.device_dirty) {
        check_cuda(
            cudaMemcpy(const_cast<void*>(host), entry.device, entry.bytes, cudaMemcpyDeviceToHost),
            "synchronize device to host");
        downloaded_bytes += entry.bytes;
        entry.device_dirty = false;
    }
}

void mark_host_modified(const void* host) {
    const auto it = entries.find(host);
    if (it != entries.end()) {
        it->second.host_dirty = true;
        it->second.host_mutable = true;
        it->second.device_dirty = false;
    }
}

void mark_device_modified(void* host) {
    const auto it = entries.find(host);
    if (it != entries.end()) {
        it->second.device_dirty = true;
        it->second.host_dirty = false;
        it->second.host_mutable = false;
    }
}

void reset_transfer_stats() noexcept {
    uploaded_bytes = 0;
    downloaded_bytes = 0;
}

TransferStats transfer_stats() noexcept {
    return TransferStats{uploaded_bytes, downloaded_bytes};
}

}  // namespace mini_gpt::cuda
