#include <sstream>
#include <libtensor/version.h>
#include "../test_utils.h"

using namespace libtensor;


int main() {

    std::ostringstream ss;
    ss << version::get_major() << "." << version::get_minor() << "-"
        << version::get_status();

    std::string ver_ref(ss.str()), ver(version::get_string());
    if(ver != ver_ref) {
        std::ostringstream sserr;
        sserr << "Version inconsistency: " << ver << " (actual) vs. "
            << ver_ref << " (ref).";
        return fail_test("version_test::perform()", __FILE__, __LINE__,
            sserr.str().c_str());
    }

    return 0;
}

