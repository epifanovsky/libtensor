
#include "allocator_wrapper.h"
#include "std_allocator.h"
#include "../allocator.h"
#include "../allocator_init.h"

#if !defined(WITHOUT_LIBVMM)
#include "vm_allocator.h"
#endif

#if defined(WITH_LIBXM)
#include "xm_allocator.h"
#endif

namespace libtensor {


struct default_allocator {
    typedef std_allocator<double> type;
};


const typename allocator::pointer_type allocator::invalid_pointer(
    allocator::make_invalid_pointer());

allocator_wrapper_i *allocator::m_aimpl(
    allocator::make_default_allocator());

size_t allocator::m_base_sz = 0;
size_t allocator::m_min_sz = 0;
size_t allocator::m_max_sz = 0;


typename allocator::pointer_type allocator::make_invalid_pointer() {

    typedef typename default_allocator::type defalloc;
    return allocator_wrapper<defalloc>::make_invalid_pointer();
}

allocator_wrapper_i *allocator::make_default_allocator() {

    typedef typename default_allocator::type defalloc;
    static allocator_wrapper<defalloc> a;
    return &a;
}


void allocator::init(size_t base_sz, size_t min_sz, size_t max_sz,
    size_t mem_limit, const char *pfprefix) {

#if defined(WITHOUT_LIBVMM)
#if defined(WITH_LIBXM)
    typedef lt_xm_allocator::lt_xm_allocator<double> allocator_impl;
#else
    typedef std_allocator<double> allocator_impl;
#endif
#else
    //typedef vm_allocator allocator_impl;
    typedef vm_allocator<double> allocator_impl;
#endif

    init(allocator_impl(), base_sz, min_sz, max_sz, mem_limit, pfprefix);
}



} // namespace libtensor

