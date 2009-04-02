#ifndef LIBTENSOR_SCALAR_H
#define LIBTENSOR_SCALAR_H

#include "defs.h"
#include "exception.h"

namespace libtensor {

/**	\brief Stores a single scalar value

	\ingroup libtensor
**/
template<typename T>
class scalar {
private:
	T m_val;

public:
	scalar(const T &t);
	T get_value();
};

template<typename T>
inline scalar<T>::scalar(const T &t) : m_val(t) {
}

template<typename T>
inline T scalar<T>::get_value() {
	return m_val;
}

} // namespace libtensor

#endif // LIBTENSOR_SCALAR_H

