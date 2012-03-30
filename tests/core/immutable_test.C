#include <libtensor/core/immutable.h>
#include "immutable_test.h"

namespace libtensor {

namespace immutable_test_ns {

class immut : public immutable {
protected:
    virtual void on_set_immutable() { }
};

}

using namespace immutable_test_ns;

void immutable_test::perform() throw(libtest::test_exception) {
    immut im;
    if(im.is_immutable()) {
        fail_test("immutable_test::perform()", __FILE__, __LINE__,
            "New object must be mutable");
    }
    im.set_immutable();
    if(!im.is_immutable()) {
        fail_test("immutable_test::perform()", __FILE__, __LINE__,
            "set_immutable() failed");
    }
}

} // namespace libtensor

