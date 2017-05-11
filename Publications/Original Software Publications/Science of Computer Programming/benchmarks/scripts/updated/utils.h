#ifndef __UTILS_H
#define __UTILS_H

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <cstdio>
#include <dirent.h>
#include <unistd.h>

#define BUFFER_SIZE 128
#define CURRENT_FOLDER_SYMLINK "."
#define PARENT_FOLDER_SYMLINK ".."
#define PIPE_READ_MODE "r"

namespace utils
{
    std::vector<std::string> execute_command(const char *command);
    std::vector<std::string> get_dirs();
}

#endif
