#ifndef BTENSOR_RENDERER_H
#define BTENSOR_RENDERER_H

#include <memory>
#include <libtensor/expr/expression_renderer_i.h>
#include "inst/btensor_operation_container_i.h"

namespace libtensor {


/** \brief Renders expressions with block tensors

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class btensor_renderer : public expression_renderer_i<N, T> {
public:
    /** \brief Virtual destructor
     **/
    virtual ~btensor_renderer() { }

    /** \brief Clones the renderer using operator new
     **/
    virtual expression_renderer_i<N, T> *clone() const {
        return new btensor_renderer<N, T>();
    }

    /** \brief Computes the expression into the output tensor
     **/
    virtual void render(const expression<N, T> &e, anytensor<N, T> &t);

    /** \brief Computes the expression into a new btensor
     **/
    std::auto_ptr< btensor_operation_container_i<N, T> > render(
        const expression<N, T> &e);

private:
    std::auto_ptr< btensor_operation_container_i<N, T> > render_node(
        expression_node<N, T> &n);

};


} // namespace libtensor

#endif // BTENSOR_RENDERER_H
