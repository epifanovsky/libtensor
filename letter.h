#ifndef LIBTENSOR_LETTER_H
#define LIBTENSOR_LETTER_H

#include "defs.h"
#include "exception.h"

namespace libtensor {

/**	/brief Identifies a letter %tensor %index

	This is an empty class that serves the purpose of identifying
	%letter indexes of a %tensor in %tensor expressions.

	Letter indexes can be combined using the multiplication (*) and the
	bitwise or (|) operators.

	\ingroup libtensor
**/
class letter {
};

} // namespace libtensor

#endif // LIBTENSOR_LETTER_H

