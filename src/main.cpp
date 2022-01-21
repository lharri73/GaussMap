#include <iostream>
#include <ros/ros.h>
#include "ecocar_fusion/gaussMapRos.hpp"

using namespace std;

int main(int argc, char** argv){
    
    ros::init(argc, argv, "gaussMap_fusion");
    ros::NodeHandle nh;
    ros::NodeHandle nh_p("~");

    GaussMapRos fusion(nh, nh_p);
    
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();
    return 0;
}