#include "core/allocator.h"
#include <utility>

namespace infini {
Allocator::Allocator(Runtime runtime) : runtime(runtime) {
  used = 0;
  peak = 0;
  ptr = nullptr;

  // 'alignment' defaults to sizeof(uint64_t), because it is the length of
  // the longest data type currently supported by the DataType field of
  // the tensor
  alignment = sizeof(uint64_t);
}

Allocator::~Allocator() {
  if (this->ptr != nullptr) {
    runtime->dealloc(this->ptr);
  }
}

size_t Allocator::alloc(size_t size) {
  IT_ASSERT(this->ptr == nullptr);
  // pad the size to the multiple of alignment
  size = this->getAlignedSize(size);

  // Find in free_blocks
  auto it = free_blocks.begin();
  while (it != free_blocks.end()) {
    const auto &[start, block_size] = *it;
    // replace
    if (block_size >= size) {
      const auto found = start;
      *it = std::make_tuple(start + size, block_size - size);
      if (block_size == size) {
        free_blocks.erase(it);
      }
      used += size;
      peak = std::max(peak, used);
      return found;
    }
    ++it;
  }

  // If last block reaches the end
  auto last_block = free_blocks.rbegin();
  const auto &[start, block_size] = *last_block;
  if (start + block_size == peak) {
    const auto found = start;
    used += size;
    peak += size - block_size;
    free_blocks.pop_back();
    return found;
  }

  // Allocate new
  const auto found = peak;
  used += size;
  peak += size;
  return found;
}

void Allocator::free(size_t addr, size_t size) {
  IT_ASSERT(this->ptr == nullptr);
  size = getAlignedSize(size);

  used -= size;

  // Find the first block that starts after the block to be freed
  auto it = free_blocks.begin();
  while (it != free_blocks.end()) {
    const auto &[start, block_size] = *it;
    if (start > addr) {
      break;
    }
    ++it;
  }

  // insert
  free_blocks.insert(it, std::make_tuple(addr, size));

  // TODO: defragmentation
}

void *Allocator::getPtr() {
  if (this->ptr == nullptr) {
    this->ptr = runtime->alloc(this->peak);
    printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
  }
  return this->ptr;
}

size_t Allocator::getAlignedSize(size_t size) {
  return ((size - 1) / this->alignment + 1) * this->alignment;
}

void Allocator::info() {
  std::cout << "Used memory: " << this->used << ", peak memory: " << this->peak
            << std::endl;
}
} // namespace infini
