from kfp import dsl
from kfp.dsl import Input, InputPath, Output, OutputPath, Dataset, Model

@dsl.container_component
def load_data(
    output_dataset_train: Output[Dataset],
    output_dataset_test: Output[Dataset]
):
    return dsl.ContainerSpec(
        image='timsmans/ml-pipeline:v11',  # Your Docker image
        command=['python', '/app/data/load_data.py'],
        args=[
            '--output_dataset_train', output_dataset_train.path,
            '--output_dataset_test', output_dataset_test.path
        ]
    )

@dsl.container_component
def define_model(
    model_output: Output[Model]
):
    return dsl.ContainerSpec(
        image='timsmans/ml-pipeline:v11',  # Your Docker image
        command=['python', '/app/define_model.py'],
        args=['--model_output', model_output.path]
    )

@dsl.container_component
def define_loss(
    model_input: Input[Model],
    loss_output: Output[Dataset],
    optimizer_output: Output[Dataset]
):
    return dsl.ContainerSpec(
        image='timsmans/ml-pipeline:v11',  # Your Docker image
        command=['python', '/app/define_loss.py'],
        args=[
            '--model_input', model_input.path,
            '--loss_output', loss_output.path,
            '--optimizer_output', optimizer_output.path
        ]
    )

@dsl.container_component
def train_model(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    model_input: Input[Model],
    loss_input: Input[Dataset],
    optimizer_input: Input[Dataset],
    trained_model: Output[Model]
):
    return dsl.ContainerSpec(
        image='timsmans/ml-pipeline:v11',  # Your Docker image
        command=['python', '/app/train_model.py'],
        args=[
            '--train_data', train_data.path,
            '--test_data', test_data.path,
            '--model_input', model_input.path,
            '--loss_input', loss_input.path,
            '--optimizer_input', optimizer_input.path,
            '--trained_model', trained_model.path
        ]
    )

@dsl.pipeline(
    name="mnist-pipeline",
    description="End-to-end MNIST training pipeline using Docker"
)
def mnist_pipeline():
    # Data loading
    load_data_task = load_data()
    
    # Model definition
    define_model_task = define_model()
    
    # Loss definition
    define_loss_task = define_loss(
        model_input=define_model_task.outputs['model_output']
    )
    
    # Training
    train_task = train_model(
        train_data=load_data_task.outputs['output_dataset_train'],
        test_data=load_data_task.outputs['output_dataset_test'],
        model_input=define_model_task.outputs['model_output'],
        loss_input=define_loss_task.outputs['loss_output'],
        optimizer_input=define_loss_task.outputs['optimizer_output']
    )

    # Define execution order
    define_loss_task.after(define_model_task)
    train_task.after(define_loss_task)

if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=mnist_pipeline,
        package_path="mnist_training_pipeline.yaml"
    )