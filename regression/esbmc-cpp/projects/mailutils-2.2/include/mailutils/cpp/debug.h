/*
   GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2009, 2010 Free Software Foundation, Inc.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General
   Public License along with this library; if not, write to the
   Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301 USA
*/

#ifndef _MUCPP_DEBUG_H
#define _MUCPP_DEBUG_H

#include <string>
#include <errno.h>
#include <mailutils/debug.h>
#include <mailutils/cpp/error.h>

namespace mailutils
{

class Debug
{
 protected:
  mu_debug_t debug;

 public:
  Debug ();
  Debug (const mu_debug_t);

  void set_level (const mu_log_level_t level);
};

}

#endif // not _MUCPP_DEBUG_H

