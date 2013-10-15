#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT3_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT3_CORE_H

#include <libtensor/core/sequence.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Three-tensor contraction operation expression core
    \tparam N1 Number of outer indices in first tensor (A).
    \tparam N2 Number of outer indices in second tensor (B).
    \tparam N3 Number of outer indices in third tensor (C).
    \tparam K1 Number of contracted indices between A and B.
    \tparam K2a Number of contracted indices between A and C.
    \tparam K2b Number of contracted indices between B and C.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2a, size_t K2b,
    typename T>
class contract3_core : public expr_core_i<N1 + N2 + N3, T> {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        K2 = K2a + K2b,
        NA = N1 + K1 + K2a,
        NB = N2 + K1 + K2b,
        NC = N3 + K2,
        ND = N1 + N2 + N3
    };

private:
    letter_expr<K1> m_contr1; //!< Contracted indexes (A*B)
    letter_expr<K2> m_contr2; //!< Contracted indexes (AB * C)
    expr_rhs<NA, T> m_expr1; //!< First expression (A)
    expr_rhs<NB, T> m_expr2; //!< Second expression (B)
    expr_rhs<NC, T> m_expr3; //!< Third expression (C)
    sequence<ND, const letter*> m_defout; //!< Default output label

public:
    /** \brief Creates the expression core
        \param contr1 Letter expression indicating which indexes will be
            contracted between A and B.
        \param contr2 Contracted indexes between AB and C.
        \param expr1 First expression (A).
        \param expr2 Second expression (B).
        \param expr3 Third expression (C).
        \throw expr_exception If letters are inconsistent.
     **/
    contract3_core(
        const letter_expr<K1> &contr1, const letter_expr<K2> &contr2,
        const expr_rhs<NA, T> &expr1, const expr_rhs<NB, T> &expr2,
        const expr_rhs<NC, T> &expr3);

    /** \brief Virtual destructor
     **/
    virtual ~contract3_core() { }

    /** \brief Clones this object using new
     **/
    virtual expr_core_i<N1 + N2 + N3, T> *clone() const {
        return new contract3_core<N1, N2, N3, K1, K2a, K2b, T>(*this);
    }

    /** \brief Returns the first expression (A)
     **/
    expr_rhs<N1 + K1 + K2a, T> &get_expr_1() {
        return m_expr1;
    }

    /** \brief Returns the first expression (A), const version
     **/
    const expr_rhs<N1 + K1 + K2a, T> &get_expr_1() const {
        return m_expr1;
    }

    /** \brief Returns the second expression (B)
     **/
    expr_rhs<N2 + K1 + K2b, T> &get_expr_2() {
        return m_expr2;
    }

    /** \brief Returns the second expression (B), const version
     **/
    const expr_rhs<N2 + K1 + K2b, T> &get_expr_2() const {
        return m_expr2;
    }

    /** \brief Returns the third expression (C)
     **/
    expr_rhs<N3 + K2a + K2b, T> &get_expr_3() {
        return m_expr3;
    }

    /** \brief Returns the third expression (C), const version
     **/
    const expr_rhs<N3 + K2a + K2b, T> &get_expr_3() const {
        return m_expr3;
    }

    /** \brief Returns contracted indexes between A and B
     **/
    const letter_expr<K1> &get_contr1() const {
        return m_contr1;
    }

    /** \brief Returns contracted indexes between AB and C
     **/
    const letter_expr<K2a + K2b> &get_contr2() const {
        return m_contr2;
    }

    /** \brief Creates evaluation container using new
     **/
    virtual eval_container_i<N1 + N2 + N3, T> *create_container(
        const letter_expr<N1 + N2 + N3> &label) const;

    /** \brief Returns whether the result's label contains a letter
        \param let Letter.
     **/
    virtual bool contains(const letter &let) const;

    /** \brief Returns the index of a letter in the result's label
        \param let Letter.
        \throw expr_exception If the label does not contain the
            requested letter.
     **/
    virtual size_t index_of(const letter &let) const;

    /** \brief Returns the letter at a given position in
            the result's label
        \param i Letter index.
        \throw out_of_bounds If the index is out of bounds.
     **/
    virtual const letter &letter_at(size_t i) const;

};


} // namespace labeled_btensor_expr
} // namespace libtensor

#include "contract_subexpr_labels.h"
#include "contract3_eval_functor.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Evaluating container for the contraction of three tensors
    \tparam N1 Number of outer indices in first tensor (A).
    \tparam N2 Number of outer indices in second tensor (B).
    \tparam N3 Number of outer indices in third tensor (C).
    \tparam K1 Number of contracted indices between A and B.
    \tparam K2a Number of contracted indices between A and C.
    \tparam K2b Number of contracted indices between B and C.

    \sa contract3_core

    \ingroup libtensor_btensor_expr
 **/
template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2a, size_t K2b,
    typename T>
class contract3_eval : public eval_container_i<N1 + N2 + N3, T> {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        K2 = K2a + K2b,
        NA = N1 + K1 + K2a,
        NB = N2 + K1 + K2b,
        NC = N3 + K2,
        ND = N1 + N2 + N3
    };

private:
    contract3_core<N1, N2, N3, K1, K2a, K2b, T> m_core; //!< Expression core
//    contract_subexpr_labels<N, M, K, T>
//        m_sub_labels; //!< Labels for sub-expressions
//    contract3_eval_functor<N1, N2, N3, K1, K2a, K2b, T>
//        m_func; //!< Sub-expression evaluation functor

public:
    /** \brief Initializes the container with given expression and
            result recipient
     **/
    contract3_eval(
        const contract3_core<N1, N2, N3, K1, K2a, K2b, T> &core,
        const letter_expr<ND> &label);

    /** \brief Virtual destructor
     **/
    virtual ~contract3_eval() { }

    /** \brief Evaluates sub-expressions into temporary tensors
     **/
    virtual void prepare();

    /** \brief Cleans up temporary tensors
     **/
    virtual void clean();

    /** \brief Returns the number of tensors in expression
     **/
    virtual size_t get_ntensor() const {
        return 0;
    }

    /** \brief Returns the number of tensor operations in expression
     **/
    virtual size_t get_noper() const {
        return 1;
    }

    /** \brief Returns tensor arguments (not valid)
        \param i Argument number.
     **/
    virtual arg<N1 + N2 + N3, T, tensor_tag> get_tensor_arg(size_t i);

    /** \brief Returns operation arguments
        \param i Argument number (0 is the only valid value).
     **/
    virtual arg<N1 + N2 + N3, T, oper_tag> get_oper_arg(size_t i);

};


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2a, size_t K2b,
    typename T>
const char contract3_core<N1, N2, N3, K1, K2a, K2b, T>::k_clazz[] =
    "contract3_core<N1, N2, N3, K1, K2a, K2b, T>";


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2a, size_t K2b,
    typename T>
contract3_core<N1, N2, N3, K1, K2a, K2b, T>::contract3_core(
    const letter_expr<K1> &contr1, const letter_expr<K2> &contr2,
    const expr_rhs<NA, T> &expr1, const expr_rhs<NB, T> &expr2,
    const expr_rhs<NC, T> &expr3) :

    m_contr1(contr1), m_contr2(contr2), m_expr1(expr1), m_expr2(expr2),
    m_expr3(expr3), m_defout(0) {

    static const char method[] = "contract3_core()";

    const expr_core_i<NA, T> &core1 = expr1.get_core();
    const expr_core_i<NB, T> &core2 = expr2.get_core();
    const expr_core_i<NC, T> &core3 = expr3.get_core();

    for(size_t i = 0; i < K1; i++) {
        const letter &l = contr1.letter_at(i);
        if(!core1.contains(l) || !core2.contains(l)) {
            throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Contracted index (1) is absent from arguments.");
        }
    }
    for(size_t i = 0; i < K2; i++) {
        const letter &l = contr2.letter_at(i);
        if(!(core1.contains(l) || core2.contains(l)) || !core3.contains(l)) {
            throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Contracted index (2) is absent from arguments.");
        }
        if(contr1.contains(l)) {
            throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Duplicate contraction indices in (1) and (2).");
        }
    }

    size_t j = 0;
    for(size_t i = 0; i < NA; i++) {
        const letter &l = core1.letter_at(i);
        if(!contr1.contains(l) && !contr2.contains(l)) {
            if(core2.contains(l) || core3.contains(l)) {
                throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Duplicate uncontracted index in A.");
            } else {
                m_defout[j++] = &l;
            }
        }
    }
    for(size_t i = 0; i < NB; i++) {
        const letter &l = core2.letter_at(i);
        if(!contr1.contains(l) && !contr2.contains(l)) {
            if(core1.contains(l) || core3.contains(l)) {
                throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Duplicate uncontracted index in B.");
            } else {
                m_defout[j++] = &l;
            }
        }
    }
    for(size_t i = 0; i < NC; i++) {
        const letter &l = core3.letter_at(i);
        if(!contr1.contains(l) && !contr2.contains(l)) {
            if(core1.contains(l) || core2.contains(l)) {
                throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Duplicate uncontracted index in C.");
            } else {
                m_defout[j++] = &l;
            }
        }
    }

    if(j != ND) {
        throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Inconsistent expression.");
    }
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2a, size_t K2b,
    typename T>
bool contract3_core<N1, N2, N3, K1, K2a, K2b, T>::contains(
    const letter &let) const {

    for(size_t i = 0; i < ND; i++) {
        if(m_defout[i] == &let) return true;
    }
    return false;
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2a, size_t K2b,
    typename T>
size_t contract3_core<N1, N2, N3, K1, K2a, K2b, T>::index_of(
    const letter &let) const {

    static const char method[] = "index_of(const letter&)";

    for(size_t i = 0; i < ND; i++) {
        if(m_defout[i] == &let) return i;
    }

    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Expression doesn't contain the letter.");
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2a, size_t K2b,
    typename T>
const letter& contract3_core<N1, N2, N3, K1, K2a, K2b, T>::letter_at(
    size_t i) const {

    static const char method[] = "letter_at(size_t)";

    if(i >= ND) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Letter index is out of bounds.");
    }
    return *(m_defout[i]);
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2a, size_t K2b,
    typename T>
const char contract3_eval<N1, N2, N3, K1, K2a, K2b, T>::k_clazz[] =
    "contract3_eval<N, M, K, T>";


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2a, size_t K2b,
    typename T>
contract3_eval<N1, N2, N3, K1, K2a, K2b, T>::contract3_eval(
    const contract3_core<N1, N2, N3, K1, K2a, K2b, T> &core,
    const letter_expr<ND> &label) :

    m_core(core)/*,
    m_sub_labels(core, label),
    m_func(m_core, m_sub_labels, label)*/ {

}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2a, size_t K2b,
    typename T>
void contract3_eval<N1, N2, N3, K1, K2a, K2b, T>::prepare() {

//    m_func.evaluate();
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2a, size_t K2b,
    typename T>
void contract3_eval<N1, N2, N3, K1, K2a, K2b, T>::clean() {

//    m_func.clean();
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2a, size_t K2b,
    typename T>
arg<N1 + N2 + N3, T, tensor_tag>
contract3_eval<N1, N2, N3, K1, K2a, K2b, T>::get_tensor_arg(size_t i) {

    static const char method[] = "get_tensor_arg(size_t)";

    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Invalid method.");
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2a, size_t K2b,
    typename T>
arg<N1 + N2 + N3, T, oper_tag>
contract3_eval<N1, N2, N3, K1, K2a, K2b, T>::get_oper_arg(size_t i) {

    static const char method[] = "get_oper_arg(size_t)";

    if(i != 0) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Argument index is out of bounds.");
    }

//    return m_func.get_arg();
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2a, size_t K2b,
    typename T>
eval_container_i<N1 + N2 + N3, T>*
contract3_core<N1, N2, N3, K1, K2a, K2b, T>::create_container(
    const letter_expr<N1 + N2 + N3> &label) const {

    return new contract3_eval<N1, N2, N3, K1, K2a, K2b, T>(*this, label);
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2, typename T,
    size_t KK>
struct contract3_core_dispatch {

    static expr<N1 + N2 + N3 - 2 * K1 - 2 * K2, T> dispatch(
        const letter_expr<K1> contr1,
        expr<N1, T> bta,
        expr<N2, T> btb,
        const letter_expr<K2> contr2,
        expr<N3, T> btc) {

        size_t kk = 0;
        for(size_t i = 0; i < K2; i++) {
            if(bta.get_core().contains(contr2.letter_at(i))) kk++;
        }
        if(kk == KK) {
            enum {
                K2a = KK,
                K2b = K2 - KK,
                M1 = N1 - K1 - K2a,
                M2 = N2 - K1 - K2b,
                M3 = N3 - K2
            };
            return expr<M1 + M2 + M3, T>(
                contract3_core<M1, M2, M3, K1, K2a, K2b, T>(
                    contr1, contr2, bta, btb, btc));
        } else {
            return contract3_core_dispatch<N1, N2, N3, K1, K2, T, KK - 1>::
                dispatch(contr1, bta, btb, contr2, btc);
        }
    }
};


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2, typename T>
struct contract3_core_dispatch<N1, N2, N3, K1, K2, T, 0> {

    static expr<N1 + N2 + N3 - 2 * K1 - 2 * K2, T> dispatch(
        const letter_expr<K1> contr1,
        expr<N1, T> bta,
        expr<N2, T> btb,
        const letter_expr<K2> contr2,
        expr<N3, T> btc) {

        enum {
            M1 = N1 - K1,
            M2 = N2 - K1 - K2,
            M3 = N3 - K2
        };
        return expr<M1 + M2 + M3, T>(
            contract3_core<M1, M2, M3, K1, 0, K2, T>(
                contr1, contr2, bta, btb, btc));
    }
};


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT3_CORE_H
