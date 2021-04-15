#pragma once
#include "ecocar_fusion/gaussMap.hpp"

#include "mobileye_560_660_msgs/ObstacleData.h"
#include "sensor_msgs/PointCloud2.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer_interface.h"
#include "geometry_msgs/TransformStamped.h"
#include "nav_msgs/OccupancyGrid.h"
#include "diagnostic_updater/diagnostic_updater.h"
#include "diagnostic_updater/publisher.h"
#include "autoware_msgs/DetectedObject.h"
#include "autoware_msgs/DetectedObjectArray.h"


class GaussMapRos : public GaussMap{
    public:
        GaussMapRos(ros::NodeHandle nh, ros::NodeHandle nh_p);
        ~GaussMapRos();
        void addRadarData(const float* array, array_info info); // popylates array with radar data using gaussian distributions
        void addCameraData(const float* array, array_info info); // adds camera data to device memory
        
        // callback function for Mobileye detections topic
        void mobileyeCallback(const mobileye_560_660_msgs::ObstacleData &msg);
        // callback function for radar detections topic
        void radarCallback(const sensor_msgs::PointCloud2 &msg);
        array_t getMobileyeData();          // returns current camera data in format used for addCameraData
        array_t getRadarData();             // returns current radar data in format used for addRadarData
        
    private:
        ros::NodeHandle nh_;                // public nodeHandle
        ros::NodeHandle nh_p_;              // private nodeHandle
        std::string mobileyeTopic_;         // MobilEye detections topic
        std::string radarTopic_;            // Radar detections topic
        std::string bboxVisTopic_;          // Bounding boxes topic used for rviz
        std::string bboxTopic_;             // Bounding boxes topic

        bool visualizeBoxes_;               // parameter to publish viewable bounding boxes and heatmap for rviz

        void cleanMeMap();                  // removes old mobileye detections
        void cleanRadMap();                 // removes old radar point clouds
        
        int _radMaxAge;                     // in ms. specefies max age of rad. PC before removal from update cycle
        int _meMaxAge;                      // in ms. specefies max age of ME det. before removal from update cycle
        float minExistProb;                 // threshold to exclude radar points. if wExist < minExistProb, exclude

        std::map<uint16_t,obstacle_t> _meData; // container for current mobileye data. indexed on obstacle id
        std::map<ros::Time,std::vector<float> > _radData; // container for current radar data. indexed on time received
        
        ros::Timer timer_;                  // fusion timer. calls runFusion
        double _refreshRate;                // Detection refresh rate. in HZ
        void runFusion(const ros::TimerEvent&); // called from timer_. calls getXXdata and calcMax. publishes results and vis if necesary

        void sendObjs(array_t fused, ros::Time start); // sends results and visualizable topics if needed
        void sendGrid();                    // sends the heat map in a nav_msgs::OccupancyGrid msg

        ros::Subscriber radSub;             // sub for radar pointclouds
        ros::Subscriber meSub;              // sub for ME obstacles
        ros::Publisher pub_bbox_vis_;       // Publisher for visualizable fusion results
        ros::Publisher pub_pose_;           // publisher for visualizable velocity
        ros::Publisher pub_bbox_;           // publisher for detection results
        ros::Publisher grid_pub_;           // publisher for visualizable heatmap

        // diagnostic publisher reports the latency of the entire algorithm as
        // well as the current update rate, should it vary
        diagnostic_updater::DiagnosedPublisher<autoware_msgs::DetectedObjectArray> *pub_bbox_freq_;
        diagnostic_updater::Updater updater_; // updater to send diagnostic events
        
        tf2_ros::Buffer tfBuffer;           // listens and records last mobileye transform from vehicle frame
        tf2_ros::TransformListener *tfListener; // listens for the transforms reported to tfBuffer. initialized in constructor

        std::mutex radMapMutex;             // prevents access to the radar map while being written
        std::mutex meMapMutex;              // prevents access to the mobieleye map while being written
};
