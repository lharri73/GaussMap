
#pragma once
/* these must be defined statically. They are used to define
 * a type whose size cannot change at runtime. */

// size of one quadrant of the search window when looking
// for local maxima. This does not include the center line itself
#define winSize 3

static_assert(winSize >= 1);

// this is the total area of the search window ((2*winSize)+1)^2
#define searchSize ((size_t)((2*winSize+1)*(2*winSize+1)))

#ifdef NUSCENES
#define ROS_WARN_STREAM(error) std::cerr << "[error] " << error << '\n';
#define ROS_ERROR_STREAM(error) std::cerr << "[warn ] " << error << '\n';
#define ROS_INFO_STREAM(info) std::cout << "[info ] " << info << '\n';
#endif
