#include "memory_reserve.h"
#include "../exception.h"

namespace libtensor {

const char* memory_reserve::k_clazz = "memory_reserve";

size_t memory_reserve::get_mem_avail() const
{
    return m_mem_avail;
}

size_t memory_reserve::get_n_tensors() const
{
    return m_n_tensors;
}

void memory_reserve::add_tensor(size_t mem)
{
    m_mem_avail -= mem;
    m_n_tensors += 1;
}

void memory_reserve::remove_tensor(size_t mem)
{
    m_mem_avail += mem;
    m_n_tensors -= 1;
}

memory_reserve::~memory_reserve()
{
    if(m_n_tensors != 0)
    {
        throw generic_exception(g_ns,k_clazz,"~memory_reserve(...)",
                __FILE__, __LINE__, "memory reserve destroyed with tensors still active - declare after tensors that use it in an object and declare it before stack allocated tensors");
    }
}

} // namespace libtensor
