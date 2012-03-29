#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include "tod_import_raw_stream.h"
#include "tod_import_raw_stream_impl.h"

namespace libtensor {

template class tod_import_raw_stream<1>;
template class tod_import_raw_stream<2>;
template class tod_import_raw_stream<3>;
template class tod_import_raw_stream<4>;
template class tod_import_raw_stream<5>;
template class tod_import_raw_stream<6>;

} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES
