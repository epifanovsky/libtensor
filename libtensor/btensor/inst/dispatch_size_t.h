#ifndef LIBTENSOR_DISPATCH_SIZE_T_H
#define LIBTENSOR_DISPATCH_SIZE_T_H

namespace libtensor {


template<size_t Nmin, size_t Nmax, template<size_t M, typename Tr> class A,
    typename Traits, typename Param>
struct dispatch_size_t {

    static bool dispatch(size_t n, Param &param);
};


template<size_t N, template<size_t M, typename Tr> class A,
    typename Traits, typename Param>
struct dispatch_size_t<N, N, A, Traits, Param> {

    static bool dispatch(size_t n, Param &param);
};


template<size_t Nmin, size_t Nmax, template<size_t M, typename Tr> class A,
    typename Traits, typename Param>
bool dispatch_size_t<Nmin, Nmax, A, Traits, Param>::dispatch(
    size_t n, Param &param) {

    if(dispatch_size_t<Nmin, Nmin, A, Traits, Param>::dispatch(n, param)) {
        return true;
    }

    if(dispatch_size_t<Nmin + 1, Nmax, A, Traits, Param>::dispatch(n, param)) {
        return true;
    }

    return false;
}


template<size_t N, template<size_t M, typename Tr> class A,
    typename Traits, typename Param>
bool dispatch_size_t<N, N, A, Traits, Param>::dispatch(size_t n, Param &param) {

    if(n != N) return false;
    A<N, Traits>::dispatch(param);
    return  true;
}


} // namespace libtensor

#endif // LIBTENSOR_DISPATCH_SIZE_T_H
