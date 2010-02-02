#ifndef LIBTENSOR_LETTER_H
#define LIBTENSOR_LETTER_H

#include "../defs.h"
#include "../exception.h"

namespace libtensor {

/**	/brief Identifies a letter %tensor %index

	This is an empty class that serves the purpose of identifying
	%letter indexes of a %tensor in %tensor expressions.

	Letter indexes can be combined using the multiplication (*) and the
	bitwise or (|) operators.

	\ingroup libtensor_iface
**/
class letter {
public:
	bool operator==(const letter &other) const;
	bool operator!=(const letter &other) const;
};

inline bool letter::operator==(const letter &other) const {
	return this == &other;
}

inline bool letter::operator!=(const letter &other) const {
	return this != &other;
}

} // namespace libtensor

#endif // LIBTENSOR_LETTER_H

