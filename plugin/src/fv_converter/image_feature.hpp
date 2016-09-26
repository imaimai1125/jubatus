	#ifndef JUBATUS_PLUGIN_FV_CONVERTER_IMAGE_FEATURE_HPP_
#define JUBATUS_PLUGIN_FV_CONVERTER_IMAGE_FEATURE_HPP_


#include <map>
#include <string>
#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>

#include "jubatus/core/fv_converter/binary_feature.hpp"
#include "jubatus/core/fv_converter/exception.hpp"

#include <iostream>


namespace jubatus{
namespace plugin{
namespace fv_converter{

using core::fv_converter::converter_exception;

class image_feature:public jubatus::core::fv_converter::binary_feature{
public:
	virtual ~image_feature(){}
	image_feature(
		float x_size = 50.0,
		float y_size = 50.0,
		const std::string& algorithm = "RGB"
		);
	void add_feature(
		const std::string& key,
		const std::string& value,
		std::vector<std::pair<std::string, float> >& ret_fv) const; 

	void dense_sampler(
		const cv::Mat mat,
		const int step,
		std::vector<cv::KeyPoint>& keypoint) const;

private:
	float x_size_;
	float y_size_;
	std::string algorithm_;
};


}  // namespace fv_converter
}  // namespace plugin
}  // namespace jubatus


extern "C" {
jubatus::plugin::fv_converter::image_feature* create(
		const std::map<std::string, std::string>& params);
std::string version();
} //extern "C"

#endif //JUBATUS_PLUGIN_FV_CONVERTER_IMAGE_FEATURE_HPP_
