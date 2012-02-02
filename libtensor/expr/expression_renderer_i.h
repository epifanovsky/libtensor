#ifndef LIBTENSOR_EXPRESSION_RENDERER_I_H
#define LIBTENSOR_EXPRESSION_RENDERER_I_H

#include "anytensor.h"
#include "expression.h"

namespace libtensor {


/** \brief Interface for expression renderers

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class expression_renderer_i {
public:
    /** \brief Virtual destructor
     **/
    virtual ~expression_renderer_i() { }

    /** \brief Clones the renderer using operator new
     **/
    virtual expression_renderer_i<N, T> *clone() const = 0;

    /** \brief Computes the expression into the output tensor
     **/
    virtual void render(const expression<N, T> &e, anytensor<N, T> &t) = 0;

};


} // namespace libtensor

#endif // LIBTENSOR_EXPRESSION_RENDERER_I_H
