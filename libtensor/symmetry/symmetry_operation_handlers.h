#ifndef LIBTENSOR_SYMMETRY_OPERATION_HANDLERS_H
#define LIBTENSOR_SYMMETRY_OPERATION_HANDLERS_H

#include "symmetry_operation_dispatcher.h"

namespace libtensor {


template<typename OperT>
class symmetry_operation_handlers_ex;


template<typename OperT>
class symmetry_operation_handlers {
public:
    static void install_handlers() {
        symmetry_operation_handlers_ex<OperT>::install_handlers();
    }
};


template<typename OperT>
class symmetry_operation_handlers_ex {
public:
    static void install_handlers() { }
};


} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_OPERATION_HANDLERS_H

