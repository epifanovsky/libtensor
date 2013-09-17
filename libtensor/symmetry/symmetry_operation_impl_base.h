#ifndef LIBTENSOR_SYMMETRY_OPERATION_IMPL_BASE_H
#define LIBTENSOR_SYMMETRY_OPERATION_IMPL_BASE_H

#include "symmetry_operation_impl_i.h"
#include "symmetry_operation_impl.h"
#include "symmetry_operation_params.h"

namespace libtensor {


/** \brief Base class for concrete %symmetry operations
    \tparam OperT Symmetry operation type.
    \tparam ElemT Symmetry element type.

    \ingroup libtensor_symmetry
 **/
template<typename OperT, typename ElemT>
class symmetry_operation_impl_base : public symmetry_operation_impl_i {
public:
    static const char *k_clazz; //!< Class name

public:
    typedef OperT operation_t;
    typedef ElemT element_t;
    typedef symmetry_operation_params<operation_t>
    symmetry_operation_params_t;

public:
    /** \brief Virtual destructor
     **/
    virtual ~symmetry_operation_impl_base() { }

    /** \brief Returns the %symmetry element class id
     **/
    virtual const char *get_id() const {
        return element_t::k_sym_type;
    }

    /** \brief Clones the implementation
     **/
    virtual symmetry_operation_impl_i *clone() const {
        return new symmetry_operation_impl<operation_t, element_t>;
    }

    /** \brief Invokes the operation
     **/
    virtual void perform(symmetry_operation_params_i &params) const;

protected:
    /** \brief Actually performs the operation
     **/
    virtual void do_perform(symmetry_operation_params_t &params) const = 0;

};


template<typename OperT, typename ElemT>
const char *symmetry_operation_impl_base<OperT, ElemT>::k_clazz =
        "symmetry_operation_impl_base<OperT, ElemT>";


template<typename OperT, typename ElemT>
void symmetry_operation_impl_base<OperT, ElemT>::perform(
        symmetry_operation_params_i &params) const {

    static const char *method = "perform(symmetry_operation_params_i&)";

    try {
        symmetry_operation_params_t &params2 =
                dynamic_cast<symmetry_operation_params_t&>(params);
        do_perform(params2);
    } catch(std::bad_cast&) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                "params: bad_cast");
    }
}


} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_OPERATION_IMPL_BASE_H
