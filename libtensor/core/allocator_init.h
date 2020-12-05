#ifndef LIBTENSOR_ALLOCATOR_INIT_H
#define LIBTENSOR_ALLOCATOR_INIT_H

#include "allocator.h"
#include "batching_policy_base.h"

namespace libtensor {


template<typename T> template<typename AllocatorImpl>
void allocator<T>::init(const AllocatorImpl &aimpl, size_t base_sz,
    size_t min_sz, size_t max_sz, size_t mem_limit, const char *pfprefix) {

    static allocator_wrapper<T, AllocatorImpl> a(aimpl);
    m_aimpl = &a;
    m_aimpl->init(base_sz, min_sz, max_sz, mem_limit, pfprefix);

    m_base_sz = base_sz;
    m_min_sz = min_sz;
    m_max_sz = max_sz;
    batching_policy_base::set_batch_size(
        mem_limit / min_sz / base_sz / base_sz / base_sz / 2);
}


template<typename T>
void allocator<T>::shutdown() {

    m_aimpl->shutdown();
    m_aimpl = make_default_allocator();
}

} // namespace libtensor

#include "allocator_init.h"
#include "impl/std_allocator.h"
#ifdef WITH_LIBXM
#include "impl/xm_allocator.h"
#endif

namespace libtensor {

template void allocator<double>::init<std_allocator<double> >(const std_allocator<double> &aimpl, size_t base_sz,
        size_t min_sz, size_t max_sz, size_t mem_limit, const char *pfprefix);

#ifdef WITH_LIBXM
template void allocator<double>::init<lt_xm_allocator::lt_xm_allocator<double> >(const lt_xm_allocator::lt_xm_allocator<double> &aimpl, size_t base_sz,
    size_t min_sz, size_t max_sz, size_t mem_limit, const char *pfprefix);
#endif


} // namespace libtensor
#endif // LIBTENSOR_ALLOCATOR_INIT_H

