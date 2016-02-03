#ifndef LIBTENSOR_EXPR_EVAL_CTF_BTENSOR_DOUBLE_CONTRACT_H
#define LIBTENSOR_EXPR_EVAL_CTF_BTENSOR_DOUBLE_CONTRACT_H

#include "../eval_ctf_btensor.h"
#include "eval_ctf_btensor_evaluator_i.h"

namespace libtensor {
namespace expr {
namespace eval_ctf_btensor_double {


template<size_t NC>
class contract : public eval_ctf_btensor_evaluator_i<NC, double> {
public:
    enum {
        Nmax = eval_ctf_btensor<double>::Nmax
    };

    typedef typename eval_ctf_btensor_evaluator_i<NC, double>::bti_traits
        bti_traits;
    typedef expr_tree::node_id_t node_id_t; //!< Node ID type

private:
    eval_ctf_btensor_evaluator_i<NC, double> *m_impl;
    bool m_add;

public:
    /** \brief Initializes the evaluator
     **/
    contract(const expr_tree &tree, node_id_t &id,
        const tensor_transf<NC, double> &tr);

    /** \brief Virtual destructor
     **/
    virtual ~contract();

    /** \brief Returns the block tensor operation
     **/
    virtual additive_gen_bto<NC, bti_traits> &get_bto() const {
        return m_impl->get_bto();
    }

};


} // namespace eval_ctf_btensor_double
} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_EVAL_CTF_BTENSOR_DOUBLE_CONTRACT_H
