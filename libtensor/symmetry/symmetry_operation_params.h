#ifndef LIBTENSOR_SYMMETRY_OPERATION_PARAMS_H
#define LIBTENSOR_SYMMETRY_OPERATION_PARAMS_H

namespace libtensor {

class symmetry_operation_params_i {
public:
    virtual ~symmetry_operation_params_i() { }
};


/** \brief Structure template for %symmetry operation parameters
    \tparam OperT Symmetry operation type.

    \ingroup libtensor_symmetry
 **/
template<typename OperT>
class symmetry_operation_params;


} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_OPERATION_PARAMS_H

