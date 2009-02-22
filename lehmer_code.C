#include "lehmer_code.h"

namespace libtensor {

lehmer_code::lehmer_code() {
}

size_t lehmer_code::perm2code(const permutation &p) throw(exception) {
	return 0;
}

const permutation &lehmer_code::code2perm(const size_t code) throw(exception) {
	throw exception("lehmer_code::code2perm: NIY");
}

} // namespace libtensor

