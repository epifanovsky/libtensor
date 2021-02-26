#ifndef LIBTENSOR_ALLOCATOR_INIT_H
#define LIBTENSOR_ALLOCATOR_INIT_H

#include "allocator.h"
#include "batching_policy_base.h"

namespace libtensor {


//template<typename T> template<typename AllocatorImpl>
template<typename AllocatorImpl>
void allocator::init(const AllocatorImpl &aimpl, size_t base_sz,
    size_t min_sz, size_t max_sz, size_t mem_limit, const char *pfprefix) {

    static allocator_wrapper<AllocatorImpl> a(aimpl);
    m_aimpl = &a;
    m_aimpl->init(base_sz, min_sz, max_sz, mem_limit, pfprefix);

    m_base_sz = base_sz;
    m_min_sz = min_sz;
    m_max_sz = max_sz;
    batching_policy_base::set_batch_size(
        mem_limit / min_sz / base_sz / base_sz / base_sz / 2);
}


void allocator::shutdown() {

    m_aimpl->shutdown();
    m_aimpl = make_default_allocator();
}


/*
template void allocator::init<vm_allocator>(const vm_allocator &aimpl, size_t base_sz,
    size_t min_sz, size_t max_sz, size_t mem_limit, const char *pfprefix);

template void allocator::init<lt_xm_allocator<double> >(const lt_xm_allocator<double> &aimpl, size_t base_sz,
    size_t min_sz, size_t max_sz, size_t mem_limit, const char *pfprefix);

template void allocator::init<std_allocator<double> >(const std_allocator<double> &aimpl, size_t base_sz,
    size_t min_sz, size_t max_sz, size_t mem_limit, const char *pfprefix);
*/

} // namespace libtensor


#include "allocator.h"
#include "allocator_init.h"
#include "impl/std_allocator.h"

#if !defined(WITHOUT_LIBVMM)
#include "impl/vm_allocator.h"
#endif

#if defined(WITH_LIBXM)
#include "impl/xm_allocator.h"
#endif



namespace libtensor {



#if !defined(WITHOUT_LIBVMM)
//template void allocator::init<vm_allocator>(const vm_allocator &aimpl, size_t base_sz,
template void allocator::init<vm_allocator<double> >(const vm_allocator<double> &aimpl, size_t base_sz,
    size_t min_sz, size_t max_sz, size_t mem_limit, const char *pfprefix);
#endif

#if defined(WITH_LIBXM)
template void allocator::init<lt_xm_allocator::lt_xm_allocator<double> >(const lt_xm_allocator::lt_xm_allocator<double> &aimpl, size_t base_sz,
    size_t min_sz, size_t max_sz, size_t mem_limit, const char *pfprefix);
#endif

template void allocator::init<std_allocator<double> >(const std_allocator<double> &aimpl, size_t base_sz,
    size_t min_sz, size_t max_sz, size_t mem_limit, const char *pfprefix);


}






#endif // LIBTENSOR_ALLOCATOR_INIT_H

