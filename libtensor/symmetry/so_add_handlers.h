#ifndef LIBTENSOR_SO_ADD_HANDLERS_H
#define LIBTENSOR_SO_ADD_HANDLERS_H

#include "symmetry_operation_dispatcher.h"
#include "symmetry_operation_handlers.h"
#include "se_perm.h"
#include "so_add_impl_perm.h"

namespace libtensor {


template<size_t N, typename T>
class symmetry_operation_handlers< so_add<N, T> > {
public:
	typedef so_add<N, T> operation_t;
	typedef symmetry_operation_dispatcher<operation_t> dispatcher_t;

public:
	static void install_handlers() {

		static bool installed = false;
		if(installed) return;

		typedef se_perm<N, T> se_perm_t;
		dispatcher_t::get_instance().register_impl(
			symmetry_operation_impl<operation_t, se_perm_t>());

		symmetry_operation_handlers_ex<operation_t>::install_handlers();

		installed = true;
	}
};


} // namespace libtensor

#endif // LIBTENSOR_SO_ADD_HANDLERS_H
