#include <ros/ros.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/convert.h>
class objStruct {
    public:
        std::string frame_id;
        ros::Time time;
        float object[4];
};  

namespace tf2{
    template <>
    inline void doTransform(const objStruct& data_in, objStruct& data_out, 
                    const geometry_msgs::TransformStamped& transform)
    {
        data_out.object[0] = data_in.object[0] + transform.transform.translation.x;
        data_out.object[1] = data_in.object[1] + transform.transform.translation.y;
        data_out.object[2] = data_in.object[2];
        data_out.object[3] = data_in.object[3];
        data_out.time = transform.header.stamp;
        data_out.frame_id = transform.header.frame_id;
    }

    template <>
    inline const ros::Time& getTimestamp(const objStruct& t){
        return t.time;
    }

    template<>
    inline const std::string& getFrameId(const objStruct &t){
        return t.frame_id;
    }
};