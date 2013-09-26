#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_ARG_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_ARG_H

#include <libtensor/core/permutation.h>
#include <libtensor/block_tensor/btod_traits.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include "../btensor_i.h"

namespace libtensor {
namespace labeled_btensor_expr {

struct tensor_tag { };
struct oper_tag { };

template<size_t N, typename T, typename Tag>
struct arg_traits { };

template<size_t N, typename T>
struct arg_traits<N, T, oper_tag> {
    //!    \brief Block tensor operation type
    typedef direct_gen_bto<N, btod_traits::bti_traits> bto_t;
};

template<size_t N>
struct arg_traits<N, double, oper_tag> {
    typedef additive_gen_bto<N, btod_traits::bti_traits> bto_t;
};

/** \brief Generic container for an expression argument

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename Tag>
class arg {
};

/** \brief Container for a %tensor expression operand

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class arg<N, T, tensor_tag> {
private:
    btensor_i<N, T> &m_bt;
    permutation<N> m_perm;
    T m_coeff;

public:
    arg(btensor_i<N, T> &bt, const permutation<N> &perm, T coeff)
    : m_bt(bt), m_perm(perm), m_coeff(coeff) {
    }

    void scale(T c) {
        m_coeff *= c;
    }

    btensor_i<N, T> &get_btensor() {
        return m_bt;
    }

    const permutation<N> &get_perm() const {
        return m_perm;
    }

    T get_coeff() const {
        return m_coeff;
    }
};

/** \brief Container for a %tensor operation expression argument

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class arg<N, T, oper_tag> {
public:
    typedef arg_traits<N, T, oper_tag> traits_t;
    typedef typename traits_t::bto_t bto_t;

private:
    bto_t &m_op; //!< Operation reference
    T m_coeff; //!< Coefficient

public:
    arg(bto_t &op, T coeff) : m_op(op), m_coeff(coeff) {
    }

    void scale(T c) {
        m_coeff *= c;
    }

    bto_t &get_operation() {
        return m_op;
    }

    T get_coeff() const {
        return m_coeff;
    }
};


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_ARG_H
