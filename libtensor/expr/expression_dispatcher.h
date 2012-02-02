#ifndef LIBTENSOR_EXPRESSION_DISPATCHER_H
#define LIBTENSOR_EXPRESSION_DISPATCHER_H

#include <map>
#include <memory>
#include <string>
#include <libutil/singleton.h>
#include "expression_renderer_i.h"
#include "expr_exception.h"

namespace libtensor {


/** \brief Dispatches the tensor expression to the appropriate registered
        renderer for calculation

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class expression_dispatcher :
    public libutil::singleton< expression_dispatcher<N, T> > {

    friend class libutil::singleton< expression_dispatcher<N, T> >;

public:
    static const char *k_clazz; //!< Class name

private:
    std::map<std::string, expression_renderer_i<N, T>*> m_renderers;

protected:
    /** \brief Protected singleton constructor
     **/
    expression_dispatcher() { }

public:
    /** \brief Destructor
     **/
    ~expression_dispatcher();

    /** \brief Registers a renderer for the given output tensor type, but
            doesn't replace the existing renderer
     **/
    void register_renderer(const std::string &ttype,
        const expression_renderer_i<N, T> &r);

    /** \brief Invokes the appropriate renderer to compute the expression into
            the output tensor
     **/
    void render(expression<N, T> &e, anytensor<N, T> &t) const;

};


template<size_t N, typename T>
const char *expression_dispatcher<N, T>::k_clazz =
    "expression_dispatcher<N, T>";


template<size_t N, typename T>
expression_dispatcher<N, T>::~expression_dispatcher() {

    for(typename std::map<std::string, expression_renderer_i<N, T>*>::
        iterator i = m_renderers.begin(); i != m_renderers.end(); ++i) {
        delete i->second;
    }
}


template<size_t N, typename T>
void expression_dispatcher<N, T>::register_renderer(const std::string &ttype,
    const expression_renderer_i<N, T> &r) {

    typename std::map<std::string, expression_renderer_i<N, T>*>::iterator i =
        m_renderers.find(ttype);
    if(i == m_renderers.end()) {
        m_renderers.insert(std::pair<std::string,
            expression_renderer_i<N, T>*>(ttype, r.clone()));
    }
}


template<size_t N, typename T>
void expression_dispatcher<N, T>::render(expression<N, T> &e,
    anytensor<N, T> &t) const {

    static const char *method = "render(expression<N, T>&, anytensor<N, T>&)";

    typename std::map<std::string, expression_renderer_i<N, T>*>::
        const_iterator i = m_renderers.find(t.get_tensor_type());
    if(i == m_renderers.end()) {
        throw expr_exception(k_clazz, method, __FILE__, __LINE__,
            "Expression renderer not found.");
    }
    std::auto_ptr< expression_renderer_i<N, T> > r(i->second->clone());
    r->render(e, t);
}


} // namespace libtensor

#endif // LIBTENSOR_EXPRESSION_DISPATCHER_H
