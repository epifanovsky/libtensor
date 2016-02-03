#ifndef LIBTENSOR_EXPR_METAPROG_H
#define LIBTENSOR_EXPR_METAPROG_H

#include <cstdlib> // for size_t
#include <libtensor/expr/eval/eval_exception.h>

namespace libtensor {
namespace expr {
namespace eval_btensor_double {


/** \brief Dispatches a single-integer-parameter template at runtime
    \tparam Nmin Minimum value of parameter.
    \tparam Nmax Maximum value of parameter.

    Given an integer value at runtime, this dispatcher will invoke
    the appropriate templated function.

    The target must define a templated dispatch() method with a single integer
    parameter.

    \ingroup libtensor_iface
 **/
template<size_t Nmin, size_t Nmax>
struct dispatch_1 {
private:
    template<size_t N> struct tag { };

public:
    template<typename Tgt>
    static void dispatch(Tgt &tgt, size_t n) {
        do_dispatch(tag<Nmin>(), tgt, n);
    }

private:
    template<typename Tgt, size_t N>
    static void do_dispatch(const tag<N>&, Tgt &tgt, size_t n) {
        if(N == n) tgt.template dispatch<N>();
        else if(N < n) do_dispatch(tag<N + 1>(), tgt, n);
        else {
            throw eval_exception(__FILE__, __LINE__,
                "libtensor::expr::eval_btensor_double",
                "dispatch_1<Nmin, Nmax>", "do_dispatch()",
                "Failure to dispatch.");
        }
    }

    template<typename Tgt>
    static void do_dispatch(const tag<Nmax + 1>&, Tgt &tgt, size_t n) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_double",
            "dispatch_1<Nmin, Nmax>", "do_dispatch()",
            "Failure to dispatch.");
    }

};


/** \brief Explicitly instantiates a single-integer-parameter template
    \tparam Nmin Minimum value of parameter.
    \tparam Nmax Maximum value of parameter.
    \tparam T Template name.

    \ingroup libtensor_iface
 **/
template<size_t Nmin, size_t Nmax, template <size_t N> class T>
struct instantiate_template_1 {
    struct dispatcher {
        template<size_t N> void dispatch() { T<N>(); }
    };
    static void instantiate(size_t n);
};

template<size_t Nmin, size_t Nmax, template <size_t N> class T>
void instantiate_template_1<Nmin, Nmax, T>::instantiate(size_t n) {
    dispatcher d;
    dispatch_1<Nmin, Nmax>::dispatch(d, n);
}


template<bool Cond, size_t A, size_t B>
struct meta_if {
    enum {
        value = A
    };
};

template<size_t A, size_t B>
struct meta_if<false, A, B> {
    enum {
        value = B
    };
};

template<size_t A, size_t B>
struct meta_min {
    enum {
        value = meta_if<(A < B), A, B>::value
    };
};

template<size_t A, size_t B>
struct meta_max {
    enum {
        value = meta_if<(A > B), A, B>::value
    };
};


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_METAPROG_H
