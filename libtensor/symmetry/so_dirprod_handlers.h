#ifndef LIBTENSOR_SO_DIRPROD_HANDLERS_H
#define LIBTENSOR_SO_DIRPROD_HANDLERS_H

#include "symmetry_operation_dispatcher.h"
#include "symmetry_operation_handlers.h"
//#include "se_label.h"
#include "se_part.h"
#include "se_perm.h"
//#include "label/so_dirprod_impl_label.h"
#include "part/so_dirprod_impl_part.h"
#include "perm/so_dirprod_impl_perm.h"

namespace libtensor {


template<size_t N, size_t M, typename T>
class symmetry_operation_handlers< so_dirprod<N, M, T> > {
public:
    typedef so_dirprod<N, M, T> operation_t;
    typedef symmetry_operation_dispatcher<operation_t> dispatcher_t;

public:
    static void install_handlers() {

        static bool installed = false;
        if(installed) return;

//        typedef se_label<N + M, T> se_label_t;
        typedef se_part<N + M, T> se_part_t;
        typedef se_perm<N + M, T> se_perm_t;

//       dispatcher_t::get_instance().register_impl(
//                symmetry_operation_impl<operation_t, se_label_t>());
        dispatcher_t::get_instance().register_impl(
                symmetry_operation_impl<operation_t, se_part_t>());
        dispatcher_t::get_instance().register_impl(
                symmetry_operation_impl<operation_t, se_perm_t>());

        symmetry_operation_handlers_ex<operation_t>::install_handlers();

        installed = true;
    }
};


} // namespace libtensor

#endif // LIBTENSOR_SO_DIRPROD_HANDLERS_H
