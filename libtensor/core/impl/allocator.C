#include "allocator_wrapper.h"
#include "std_allocator.h"
#ifdef WITH_LIBXM
#include "xm_allocator.h"
#endif


namespace libtensor {

#ifdef WITH_LIBXM
namespace {
template <typename T>
allocator_wrapper<T, lt_xm_allocator::lt_xm_allocator<T>>* make_xm_allocator() {
    static allocator_wrapper<T, lt_xm_allocator::lt_xm_allocator<T>> a;
    return &a;
}
}
#endif

template<typename T>
const typename allocator<T>::pointer_type allocator<T>::invalid_pointer(
    allocator<T>::make_invalid_pointer());

template<typename T>
allocator_wrapper_i<T> *allocator<T>::m_aimpl(
    allocator<T>::make_default_allocator());


template<typename T>
typename allocator<T>::pointer_type allocator<T>::make_invalid_pointer() {
    return allocator_wrapper<T, std_allocator<T>>::make_invalid_pointer();
}

template<typename T>
allocator_wrapper_i<T> *allocator<T>::make_default_allocator() {
    static allocator_wrapper<T, std_allocator<T>> a;
    return &a;
}

template<typename T>
void allocator<T>::init(const std::string& allocator, const char *pfprefix) {

#ifdef WITH_LIBXM
    if (allocator == "libxm") {
        m_aimpl = make_xm_allocator<T>();
    } else
#endif
    {
        // Fall-back to default allocator
        (void) allocator;  // Fake-use argument.
        m_aimpl = make_default_allocator();
    }

    m_aimpl->init(pfprefix);
}

template<typename T>
void allocator<T>::shutdown() {
    m_aimpl->shutdown();
    m_aimpl = make_default_allocator();
}

//
// Explicit instantiation
//
template class allocator<int>;
template class allocator<double>;

} // namespace libtensor
