#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include "btod_contract2.h"
#include "btod_contract2_impl.h"

namespace libtensor {


template<>
struct btod_contract2_clazz<0, 1, 1> {
	static const char *k_clazz;
};
template<> 
struct btod_contract2_clazz<0, 1, 2> {
	static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<0, 1, 3> {
	static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<0, 1, 4> {
	static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<0, 1, 5> {
	static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<1, 0, 1> {
	static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<1, 0, 2> {
	static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<1, 0, 3> {
	static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<1, 0, 4> {
	static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<1, 0, 5> {
	static const char *k_clazz;
};


const char *btod_contract2_clazz<0, 1, 1>::k_clazz = "btod_contract2<0, 1, 1>";
const char *btod_contract2_clazz<0, 1, 2>::k_clazz = "btod_contract2<0, 1, 2>";
const char *btod_contract2_clazz<0, 1, 3>::k_clazz = "btod_contract2<0, 1, 3>";
const char *btod_contract2_clazz<0, 1, 4>::k_clazz = "btod_contract2<0, 1, 4>";
const char *btod_contract2_clazz<0, 1, 5>::k_clazz = "btod_contract2<0, 1, 5>";
const char *btod_contract2_clazz<1, 0, 1>::k_clazz = "btod_contract2<1, 0, 1>";
const char *btod_contract2_clazz<1, 0, 2>::k_clazz = "btod_contract2<1, 0, 2>";
const char *btod_contract2_clazz<1, 0, 3>::k_clazz = "btod_contract2<1, 0, 3>";
const char *btod_contract2_clazz<1, 0, 4>::k_clazz = "btod_contract2<1, 0, 4>";
const char *btod_contract2_clazz<1, 0, 5>::k_clazz = "btod_contract2<1, 0, 5>";


template class btod_contract2<0, 1, 1>;
template class btod_contract2<0, 1, 2>;
template class btod_contract2<0, 1, 3>;
template class btod_contract2<0, 1, 4>;
template class btod_contract2<0, 1, 5>;
template class btod_contract2<1, 0, 1>;
template class btod_contract2<1, 0, 2>;
template class btod_contract2<1, 0, 3>;
template class btod_contract2<1, 0, 4>;
template class btod_contract2<1, 0, 5>;


} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES
