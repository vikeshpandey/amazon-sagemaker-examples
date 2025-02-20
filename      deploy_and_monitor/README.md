# Amazon SageMaker Examples

### Deploy and Monitor

The example notebooks within this folder showcase the capabilities of Amazon SageMaker in deploying and monitoring machine learning models.

- [Deploy Models with ModelBuilder using IN_PROCESS Mode](sm-model_builder/model_builder_in_process_mode.ipynb)
- [Get Started Building and Deploying Models with ModelBuilder](sm-model_builder/model_builder_handshake.ipynb)
- [A/B Testing with Amazon SageMaker](sm-a_b_testing/sm-a_b_testing.ipynb)
- [Faster autoscaling on Amazon SageMaker realtime endpoints (Application Autoscaling)](sm-app_autoscaling_realtime_endpoints/sm-app_autoscaling_realtime_endpoints.ipynb)
- [Faster autoscaling on Amazon SageMaker realtime endpoints with inference components (Application Autoscaling)](sm-app_autoscaling_realtime_endpoints_inference_components/sm-app_autoscaling_realtime_endpoints_inference_components.ipynb)
- [Faster autoscaling on Amazon SageMaker realtime endpoints (Step Scaling)](sm-app_autoscaling_realtime_endpoints_step_scaling/sm-app_autoscaling_realtime_endpoints_step_scaling.ipynb)
- [Amazon SageMaker Asynchronous Inference](sm-async_inference_walkthrough/sm-async_inference_walkthrough.ipynb)
- [Amazon SageMaker Asynchronous Inference using the SageMaker Python SDK](sm-async_inference_with_python_sdk/sm-async_inference_with_python_sdk.ipynb)
- [SageMaker Real-time Dynamic Batching Inference with Torchserve](sm-batch_inference_with_torchserve/sm-batch_inference_with_torchserve.ipynb)
- [Amazon SageMaker Batch Transform](sm-batch_transform_pca_dbscan_movie_clusters/sm-batch_transform_pca_dbscan_movie_clusters.ipynb)
- [Use SageMaker Batch Transform for PyTorch Batch Inference](sm-batch_transform_pytorch/sm-batch_transform_pytorch.ipynb)
- [SageMaker Batch Transform with Torchserve](sm-batch_transform_with_torchserve/sm-batch_transform_with_torchserve.ipynb)
- [Amazon SageMaker Clarify Model Explainability Monitor for Batch Transform - JSON Lines Format](sm-clarify_model_bias_monitor_batch_transform/sm-clarify_model_bias_monitor_batch_transform.ipynb)
- [Amazon SageMaker Clarify Model Bias Monitor for Batch Transform - JSON Format](sm-clarify_model_bias_monitor_batch_transform_json/sm-clarify_model_bias_monitor_batch_transform_json.ipynb)
- [Amazon SageMaker Clarify Model Bias Monitor - JSON Lines Format](sm-clarify_model_bias_monitor_for_endpoint/sm-clarify_model_bias_monitor_for_endpoint.ipynb)
- [Amazon SageMaker Clarify Model Bias Monitor - JSON Format](sm-clarify_model_bias_monitor_for_endpoint_json/sm-clarify_model_bias_monitor_for_endpoint_json.ipynb)
- [Leverage deployment guardrails to update a SageMaker Inference endpoint using linear traffic shifting](sm-deployment_guardrails_update_inference_endpoint_with_linear_traffic_shifting/sm-deployment_guardrails_update_inference_endpoint_with_linear_traffic_shifting.ipynb)
- [Leverage deployment guardrails to update a SageMaker Inference endpoint using rolling deployment](sm-deployment_guardrails_update_inference_endpoint_with_rolling_deployment/sm-deployment_guardrails_update_inference_endpoint_with_rolling_deployment.ipynb)
- [Leverage deployment guardrails to update a SageMaker Inference endpoint using canary traffic shifting](sm-deployment_guardrails_update_inference_endpoint_with_with_canary_traffic_shifting/sm-deployment_guardrails_update_inference_endpoint_with_with_canary_traffic_shifting.ipynb)
- [Host a Pretrained Model on SageMaker](sm-host_pretrained_model_bert/sm-host_pretrained_model_bert.ipynb)
- [Inference Pipeline with Scikit-learn and Linear Learner](sm-inference_pipeline_with_scikit_linear_learner/sm-inference_pipeline_with_scikit_linear_learner.ipynb)
- [Amazon SageMaker Cross Account Lineage Queries](sm-lineage_cross_account_queries_with_ram/sm-lineage_cross_account_queries_with_ram.ipynb)
- [AWS Marketplace Product Usage Demonstration - Model Packages](sm-marketplace_using_model_package_arn/sm-marketplace_using_model_package_arn.ipynb)
- [Amazon SageMaker Multi-Model Endpoints using TorchServe](sm-mme_with_torchserve/sm-mme_with_torchserve.ipynb)
- [SageMaker Model Monitor with Batch Transform - Data Quality Monitoring On-Schedule](sm-model_monitor_batch_transform_data_quality_on_schedule/sm-model_monitor_batch_transform_data_quality_on_schedule.ipynb)
- [SageMaker Model Monitor with Batch Transform - Model Quality Monitoring On-schedule](sm-model_monitor_batch_transform_model_quality_on_schedule/sm-model_monitor_batch_transform_model_quality_on_schedule.ipynb)
- [Amazon SageMaker Clarify Model Monitors](sm-model_monitor_bias_and_explainability_monitoring/sm-model_monitor_bias_and_explainability_monitoring.ipynb)
- [BYOC LLM Monitoring: Bring Your Own Container Llama2 Multiple Evaluations Monitoring with SageMaker Model Monitor](sm-model_monitor_byoc_llm_monitor/sm-model_monitor_byoc_llm_monitor.ipynb)
- [Amazon SageMaker Model Monitor](sm-model_monitor_introduction/sm-model_monitor_introduction.ipynb)
- [Amazon SageMaker Model Quality Monitor](sm-model_monitor_model_quality_monitoring/sm-model_monitor_model_quality_monitoring.ipynb)
- [Running multi-container endpoints on Amazon SageMaker](sm-multi_container_endpoint_direct_invocation/sm-multi_container_endpoint_direct_invocation.ipynb)
- [Amazon SageMaker Multi-Model Endpoints using your own algorithm container](sm-multi_model_endpoint_bring_your_own_container/sm-multi_model_endpoint_bring_your_own_container.ipynb)
- [SageMaker Serverless Inference](sm-serverless_inference_huggingface_text_classification/sm-serverless_inference_huggingface_text_classification.ipynb)
- [Shadow Variant Experiments via API](sm-shadow_variant_shadow_api/sm-shadow_variant_shadow_api.ipynb)
- [Triton on SageMaker - Deploying on Inferentia instance type](sm-triton_inferentia2/sm-triton_inferentia2.ipynb)
- [Run Multiple NLP Bert Models on GPU with Amazon SageMaker Multi-Model Endpoints (MME)](sm-triton_mme_bert_trt/sm-triton_mme_bert_trt.ipynb)
- [Multiple Ensembles with GPU models using Amazon SageMaker in MME mode](sm-triton_mme_gpu_ensemble_dali/sm-triton_mme_gpu_ensemble_dali.ipynb)
- [Triton on SageMaker - NLP Bert](sm-triton_nlp_bert/sm-triton_nlp_bert.ipynb)
- [Serve Pytorch models with the Python Backend on GPU with Amazon SageMaker Hosting](sm-triton_realtime_sme_flan_t5/sm-triton_realtime_sme_flan_t5.ipynb)
- [Triton TensorRT Sentence Transformer](sm-triton_tensorrt-sentence_transformer/sm-triton_tensorrt-sentence_transformer.ipynb)
- [Amazon SageMaker XGBoost Bring Your Own Model](sm-xgboost_bring_your_own_model/sm-xgboost_bring_your_own_model.ipynb)
- [SageMaker Inference Recommender](sm-inference_recommender_introduction.ipynb)
- [SageMaker Serverless Inference](sm-serverless_inference.ipynb)
- [Implement a SageMaker Real-time Single Model Endpoint (SME) for a TensorFlow Vision model on an NVIDIA Triton Server](sm-triton_realtime_sme.ipynb)
- [Deploy a TensorFlow Model using NVIDIA Triton on SageMaker](sm-triton_tensorflow_model_deploy.ipynb)
