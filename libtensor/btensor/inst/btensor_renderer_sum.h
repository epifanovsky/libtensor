#ifndef LIBTENSOR_BTENSOR_RENDERER_SUM_H
#define LIBTENSOR_BTENSOR_RENDERER_SUM_H

#include <memory>
#include "btensor_operation_container_i.h"

namespace libtensor {


/** \brief Contains rendered sum of expressions

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class btensor_operation_container_sum;


/** \brief Contains rendered sum of expressions (specialized for double)

    \ingroup libtensor_expr
 **/
template<size_t N>
class btensor_operation_container_sum<N, double> :
    public btensor_operation_container_i<N, double> {

private:
    btensor_operation_container_list<N, double> m_ops;

public:
    /** \brief Initializes the container
     **/
    btensor_operation_container_sum() { }

    /** \brief Virtual destructor
     **/
    virtual ~btensor_operation_container_sum() { }

    /** \brief Performs the block tensor operation into the given tensor
     **/
    virtual void perform(bool add, btensor_i<N, double> &bt) {

        for(size_t i = 0; i < m_ops.size(); i++) {
            m_ops[i].perform(i || add, bt);
        }
    }

    /** \brief Performs the block tensor operation into a new tensor
     **/
    virtual std::auto_ptr< btensor_i<N, double> > perform() {

        if(m_ops.size() == 0) {
            throw 0;
        }

        std::auto_ptr< btensor_i<N, double> > bt = m_ops[0].perform();
        for(size_t i = 1; i < m_ops.size(); i++) {
            m_ops[i].perform(true, *bt);
        }
        return bt;
    }

    void add_op(std::auto_ptr< btensor_operation_container_i<N, double> > &op) {

        m_ops.push_back(op.release());
    }

};


} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_RENDERER_SUM_H
