#include "metadata.h"
#include <algorithm>
#include <sstream>
#include <vector>
#ifndef BLA_VENDOR
#define BLA_VENDOR "unknown"
#endif

namespace libtensor {
namespace {
static const std::string static_version_string = "3.0.0";

static const std::vector<std::string> version_split = [](const std::string& in) {
  std::vector<std::string> parts;
  std::stringstream ss(in);
  std::string item;
  while (std::getline(ss, item, '.')) parts.push_back(item);
  return parts;
}(static_version_string);

static int get_version_part(size_t part) {
  int ret;
  std::stringstream ss(version_split[part]);
  ss >> ret;
  return ret;
}

}  // namespace

int metadata::major_part() { return get_version_part(0); }
int metadata::minor_part() { return get_version_part(1); }
int metadata::patch_part() { return get_version_part(2); }
bool metadata::is_debug() {
#ifdef NDEBUG
  return false;
#else
  return true;
#endif  // NDEBUG
}

std::string metadata::version_string() { return static_version_string; }

std::vector<std::string> metadata::features() {
  std::vector<std::string> ret;
#ifdef WITH_LIBXM
  ret.push_back("libxm");
#endif
#ifdef WITH_MPI
  ret.push_back("mpi");
#endif

  std::sort(ret.begin(), ret.end());
  return ret;
}

std::string metadata::blas() { return std::string(BLA_VENDOR); }

std::string metadata::authors() {
  return "Evgeny Epifanovsky, Michael Wormit, Dmitry Zuev Sam Manzer, Ilya Kaliman, "
         "Michael F. Herbst and Maximilian Scheurer";
}

}  // namespace libtensor
