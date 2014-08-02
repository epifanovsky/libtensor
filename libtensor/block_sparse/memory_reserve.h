#ifndef MEMORY_RESERVE_H
#define MEMORY_RESERVE_H

#include <vector>

namespace libtensor {

class memory_reserve
{
private:
    static const char *k_clazz; //!< Class name
    size_t m_mem_avail;
    size_t m_n_tensors;
public:
    memory_reserve(size_t mem_total) : m_mem_avail(mem_total),m_n_tensors(0) {}
    size_t get_mem_avail() const;
    size_t get_n_tensors() const;
    void add_tensor(size_t mem);
    void remove_tensor(size_t mem);
    ~memory_reserve(); 
};

} // namespace libtensor

#endif /* MEMORY_RESERVE_H */
