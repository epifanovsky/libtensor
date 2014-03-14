#ifndef LIBTENSOR_EXPR_EVAL_CTF_BTENSOR_DOUBLE_DIAG_H
#define LIBTENSOR_EXPR_EVAL_CTF_BTENSOR_DOUBLE_DIAG_H

#include "../eval_ctf_btensor.h"
#include "eval_ctf_btensor_evaluator_i.h"

namespace libtensor {
namespace expr {
namespace eval_ctf_btensor_double {


template<size_t N>
class diag : public eval_ctf_btensor_evaluator_i<N, double> {
public:
    enum {
        Nmax = eval_ctf_btensor<double>::Nmax
    };

    typedef typename eval_ctf_btensor_evaluator_i<N, double>::bti_traits
        bti_traits;
    typedef expr_tree::node_id_t node_id_t; //!< Node ID type

private:
    eval_ctf_btensor_evaluator_i<N, double> *m_impl;

public:
    /** \brief Initializes the evaluator
     **/
    diag(const expr_tree &tree, node_id_t &id,
        const tensor_transf<N, double> &tr);

    /** \brief Virtual destructor
     **/
    virtual ~diag();

    /** \brief Returns the block tensor operation
     **/
    virtual additive_gen_bto<N, bti_traits> &get_bto() const {
        return m_impl->get_bto();
    }

};


} // namespace eval_ctf_btensor_double
} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_EVAL_CTF_BTENSOR_DOUBLE_DIAG_H
