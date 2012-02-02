#ifndef LIBTENSOR_BTENSOR_RENDERER_IDENT_H
#define LIBTENSOR_BTENSOR_RENDERER_IDENT_H

#include <libtensor/btod/btod_copy.h>
#include <libtensor/btensor/btensor.h>

namespace libtensor {


/** \brief Contains rendered expression_node_ident

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class btensor_operation_container_ident;


/** \brief Renders expression_node_ident into btensor_i

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class btensor_renderer_ident;


/** \brief Contains rendered expression_node_ident (specialized for double)

    \ingroup libtensor_expr
 **/
template<size_t N>
class btensor_operation_container_ident<N, double> :
    public btensor_operation_container_i<N, double> {

private:
    btensor_i<N, double> &m_bt;
    permutation<N> m_perm;
    double m_s;

public:
    /** \brief Puts a copy operation in the container (must be created via new)
     **/
    btensor_operation_container_ident(btensor_i<N, double> &bt,
        const permutation<N> &perm, double s) :

        m_bt(bt), m_perm(perm), m_s(s) { }

    /** \brief Destroys the container
     **/
    virtual ~btensor_operation_container_ident() { }

    /** \brief Performs the block tensor operation into the given tensor
     **/
    virtual void perform(bool add, btensor_i<N, double> &bt) {

        btod_copy<N> op(m_bt, m_perm, m_s);
        if(add) op.perform(bt, 1.0);
        else op.perform(bt);
    }

    /** \brief Performs the block tensor operation into a new tensor
     **/
    virtual std::auto_ptr< btensor_i<N, double> > perform() {

        btod_copy<N> op(m_bt, m_perm, m_s);
        std::auto_ptr< btensor_i<N, double> > bt(
            new btensor<N, double>(op.get_bis()));
        op.perform(*bt);
        return bt;
    }

};


/** \brief Renders expression_node_ident into btensor_i (specialized for
        double)

    \ingroup libtensor_expr
 **/
template<size_t N>
class btensor_renderer_ident<N, double> {
public:
    std::auto_ptr< btensor_operation_container_i<N, double> > render_node(
        expression_node_ident<N, double> &n) {

        btensor_i<N, double> &bt =
            dynamic_cast<btensor_i<N, double>&>(n.get_tensor());

        return std::auto_ptr< btensor_operation_container_i<N, double> >(
            new btensor_operation_container_ident<N, double>(
                bt, n.get_perm(), n.get_s()));
    }

};


} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_RENDERER_IDENT_H
