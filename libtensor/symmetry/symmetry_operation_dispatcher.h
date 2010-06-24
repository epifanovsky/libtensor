#ifndef LIBTENSOR_SYMMETRY_OPERATION_DISPATCHER_H
#define LIBTENSOR_SYMMETRY_OPERATION_DISPATCHER_H

#include <map>
#include <string>
#include <libvmm/singleton.h>
#include "../defs.h" // for g_ns
#include "../not_implemented.h"
#include "symmetry_operation_impl_i.h"

namespace libtensor {


template<typename OperT>
class symmetry_operation_dispatcher :
	public libvmm::singleton< symmetry_operation_dispatcher<OperT> > {

	friend class libvmm::singleton< symmetry_operation_dispatcher<OperT> >;

public:
	static const char *k_clazz; //!< Class name

private:
	std::map<std::string, symmetry_operation_impl_i*> m_map;

public:
	void register_impl(const symmetry_operation_impl_i &impl);

	void invoke(const std::string &id,
		symmetry_operation_params<OperT> &params);

protected:
	symmetry_operation_dispatcher() { }

};


template<typename OperT>
const char *symmetry_operation_dispatcher<OperT>::k_clazz =
	"symmetry_operation_dispatcher<OperT>";


template<typename OperT>
void symmetry_operation_dispatcher<OperT>::register_impl(
	const symmetry_operation_impl_i &impl) {

	const std::string &id = impl.get_id();
	typename std::map<std::string, symmetry_operation_impl_i*>::iterator i =
		m_map.find(id);
	if(i != m_map.end()) {
		delete i->second;
		i->second = impl.clone();
	} else {
		m_map.insert(std::pair<std::string, symmetry_operation_impl_i*>(
			id, impl.clone()));
	}
}


template<typename OperT>
void symmetry_operation_dispatcher<OperT>::invoke(const std::string &id,
	symmetry_operation_params<OperT> &params) {

	static const char *method =
		"invoke(const std::string&, symmetry_operation_params<OperT>&)";

	typename std::map<std::string, symmetry_operation_impl_i*>::iterator i =
		m_map.find(id);
#ifdef LIBTENSOR_DEBUG
	if(i == m_map.end()) {
		throw not_implemented(
			g_ns, k_clazz, method, __FILE__, __LINE__);
	}
#endif // LIBTENSOR_DEBUG
	if(i != m_map.end()) i->second->perform(params);
}


} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_OPERATION_DISPATCHER_H

