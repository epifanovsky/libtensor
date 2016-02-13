#ifndef LIBTENSOR_EXPR_EVAL_CTF_BTENSOR_DOUBLE_CONVERT_H
#define LIBTENSOR_EXPR_EVAL_CTF_BTENSOR_DOUBLE_CONVERT_H

#include "../eval_ctf_btensor.h"

namespace libtensor {
namespace expr {
namespace eval_ctf_btensor_double {


template<size_t N>
class convert {
public:
    enum {
        Nmax = eval_ctf_btensor<double>::Nmax
    };

    typedef expr_tree::node_id_t node_id_t; //!< Node ID type

private:
    const expr_tree &m_tree;
    node_id_t m_rhs;

public:
    /** \brief Initializes the evaluator
     **/
    convert(const expr_tree &tree, node_id_t &id);

    /** \brief Evaluates the result into given node
     **/
    void evaluate(node_id_t lhs);

};


} // namespace eval_ctf_btensor_double
} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_EVAL_CTF_BTENSOR_DOUBLE_CONVERT_H
