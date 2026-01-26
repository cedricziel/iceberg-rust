//! Arrow-based partitioning implementation for Iceberg tables
//!
//! This module provides functionality to partition Arrow record batches according to Iceberg partition
//! specifications. It includes:
//!
//! * Streaming partition implementation that processes record batches asynchronously
//! * Support for different partition transforms (identity, bucket, truncate)
//! * Efficient handling of distinct partition values
//! * Automatic management of partition streams and channels

use std::{collections::HashSet, hash::Hash};

use arrow::{
    array::{
        as_primitive_array, as_string_array, ArrayRef, BooleanArray, BooleanBufferBuilder,
        PrimitiveArray, StringArray,
    },
    compute::{
        and, filter, filter_record_batch,
        kernels::cmp::{distinct, eq},
    },
    datatypes::{ArrowPrimitiveType, DataType, Date32Type, Int32Type, Int64Type},
    error::ArrowError,
    record_batch::RecordBatch,
};
use itertools::{iproduct, Itertools};

use iceberg_rust_spec::{partition::BoundPartitionField, spec::values::Value};

use super::transform::transform_arrow;

/// Partitions a record batch according to the given partition fields.
///
/// This function takes a record batch and partition field specifications, then splits the batch into
/// multiple record batches based on unique combinations of partition values.
///
/// # Arguments
/// * `record_batch` - The input record batch to partition
/// * `partition_fields` - The partition field specifications that define how to split the data
///
/// # Returns
/// An iterator over results containing:
/// * A vector of partition values that identify the partition
/// * The record batch containing only rows matching those partition values
///
/// # Errors
/// Returns an ArrowError if:
/// * Required columns are missing from the record batch
/// * Transformation operations fail
/// * Data type conversions fail
pub fn partition_record_batch<'a>(
    record_batch: &'a RecordBatch,
    partition_fields: &[BoundPartitionField<'_>],
) -> Result<impl Iterator<Item = Result<(Vec<Value>, RecordBatch), ArrowError>> + 'a, ArrowError> {
    let partition_columns: Vec<ArrayRef> = partition_fields
        .iter()
        .map(|field| {
            let array = record_batch
                .column_by_name(field.source_name())
                .ok_or(ArrowError::SchemaError("Column doesn't exist".to_string()))?;
            transform_arrow(array.clone(), field.transform())
        })
        .collect::<Result<_, ArrowError>>()?;
    let distinct_values: Vec<DistinctValues> = partition_columns
        .iter()
        .map(|x| distinct_values(x.clone()))
        .collect::<Result<Vec<_>, ArrowError>>()?;
    let mut true_buffer = BooleanBufferBuilder::new(record_batch.num_rows());
    true_buffer.append_n(record_batch.num_rows(), true);
    let predicates = distinct_values
        .into_iter()
        .zip(partition_columns.iter())
        .map(|(distinct, value)| match distinct {
            DistinctValues::Int(set) => set
                .into_iter()
                .map(|x| {
                    Ok((
                        Value::Int(x),
                        eq(&PrimitiveArray::<Int32Type>::new_scalar(x), value)?,
                    ))
                })
                .collect::<Result<Vec<_>, ArrowError>>(),
            DistinctValues::Long(set) => set
                .into_iter()
                .map(|x| {
                    Ok((
                        Value::LongInt(x),
                        eq(&PrimitiveArray::<Int64Type>::new_scalar(x), value)?,
                    ))
                })
                .collect::<Result<Vec<_>, ArrowError>>(),
            DistinctValues::Date(set) => set
                .into_iter()
                .map(|x| {
                    Ok((
                        Value::Date(x),
                        eq(&PrimitiveArray::<Date32Type>::new_scalar(x), value)?,
                    ))
                })
                .collect::<Result<Vec<_>, ArrowError>>(),
            DistinctValues::String(set) => set
                .into_iter()
                .map(|x| {
                    let res = eq(&StringArray::new_scalar(&x), value)?;
                    Ok((Value::String(x), res))
                })
                .collect::<Result<Vec<_>, ArrowError>>(),
        })
        .try_fold(
            vec![(vec![], BooleanArray::new(true_buffer.finish(), None))],
            |acc, predicates| {
                iproduct!(acc, predicates?.iter())
                    .map(|((mut values, x), (value, y))| {
                        values.push(value.clone());
                        Ok((values, and(&x, y)?))
                    })
                    .filter_ok(|x| x.1.true_count() != 0)
                    .collect::<Result<Vec<(Vec<Value>, _)>, ArrowError>>()
            },
        )?;
    Ok(predicates.into_iter().map(move |(values, predicate)| {
        Ok((values, filter_record_batch(record_batch, &predicate)?))
    }))
}

/// Extracts distinct values from an Arrow array into a DistinctValues enum
///
/// # Arguments
/// * `array` - The Arrow array to extract distinct values from
///
/// # Returns
/// * `Ok(DistinctValues)` - An enum containing a HashSet of the distinct values
/// * `Err(ArrowError)` - If the array's data type is not supported
///
/// # Supported Data Types
/// * Int32 - Converted to DistinctValues::Int
/// * Int64 - Converted to DistinctValues::Long
/// * Date32 - Converted to DistinctValues::Date
/// * Utf8 - Converted to DistinctValues::String
fn distinct_values(array: ArrayRef) -> Result<DistinctValues, ArrowError> {
    match array.data_type() {
        DataType::Int32 => Ok(DistinctValues::Int(distinct_values_primitive::<
            i32,
            Int32Type,
        >(array)?)),
        DataType::Int64 => Ok(DistinctValues::Long(distinct_values_primitive::<
            i64,
            Int64Type,
        >(array)?)),
        DataType::Date32 => Ok(DistinctValues::Date(distinct_values_primitive::<
            i32,
            Date32Type,
        >(array)?)),
        DataType::Utf8 => Ok(DistinctValues::String(distinct_values_string(array)?)),
        _ => Err(ArrowError::ComputeError(
            "Datatype not supported for transform.".to_string(),
        )),
    }
}

/// Extracts distinct primitive values from an Arrow array into a HashSet
///
/// # Type Parameters
/// * `T` - The Rust native type that implements Eq + Hash
/// * `P` - The Arrow primitive type corresponding to T
///
/// # Arguments
/// * `array` - The Arrow array to extract distinct values from
///
/// # Returns
/// A HashSet containing all unique values from the array
fn distinct_values_primitive<T: Eq + Hash, P: ArrowPrimitiveType<Native = T>>(
    array: ArrayRef,
) -> Result<HashSet<P::Native>, ArrowError> {
    let array = as_primitive_array::<P>(&array);

    let first = array.value(0);

    let slice_len = array.len() - 1;

    if slice_len == 0 {
        return Ok(HashSet::from_iter([first]));
    }

    let v1 = array.slice(0, slice_len);
    let v2 = array.slice(1, slice_len);

    // Which consecutive entries are different
    let mask = distinct(&v1, &v2)?;

    let unique = filter(&v2, &mask)?;

    let unique = as_primitive_array::<P>(&unique);

    let set = unique
        .iter()
        .fold(HashSet::from_iter([first]), |mut acc, x| {
            if let Some(x) = x {
                acc.insert(x);
            }
            acc
        });
    Ok(set)
}

/// Extracts distinct string values from an Arrow array into a HashSet
///
/// # Arguments
/// * `array` - The Arrow array to extract distinct values from
///
/// # Returns
/// A HashSet containing all unique string values from the array
fn distinct_values_string(array: ArrayRef) -> Result<HashSet<String>, ArrowError> {
    let slice_len = array.len() - 1;

    let array = as_string_array(&array);

    let first = array.value(0).to_owned();

    if slice_len == 0 {
        return Ok(HashSet::from_iter([first]));
    }

    let v1 = array.slice(0, slice_len);
    let v2 = array.slice(1, slice_len);

    // Which consecutive entries are different
    let mask = distinct(&v1, &v2)?;

    let unique = filter(&v2, &mask)?;

    let unique = as_string_array(&unique);

    let set = unique
        .iter()
        .fold(HashSet::from_iter([first]), |mut acc, x| {
            if let Some(x) = x {
                acc.insert(x.to_owned());
            }
            acc
        });
    Ok(set)
}

/// Represents distinct values found in Arrow arrays during partitioning
///
/// This enum stores unique values from different Arrow array types:
/// * `Int` - Distinct 32-bit integer values
/// * `Long` - Distinct 64-bit integer values  
/// * `Date` - Distinct date values (days since Unix epoch as i32)
/// * `String` - Distinct string values
enum DistinctValues {
    Int(HashSet<i32>),
    Long(HashSet<i64>),
    Date(HashSet<i32>),
    String(HashSet<String>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use arrow::array::{Date32Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use iceberg_rust_spec::spec::partition::{PartitionField, Transform};
    use iceberg_rust_spec::spec::types::{PrimitiveType, StructField, Type};

    #[test]
    fn test_identity_transform_int32() {
        // Create a record batch with Int32 column
        let schema = ArrowSchema::new(vec![
            Field::new("int_col", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]);

        let int_array = Arc::new(Int32Array::from(vec![100, 100, 200])) as ArrayRef;
        let value_array = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;

        let record_batch =
            RecordBatch::try_new(Arc::new(schema), vec![int_array, value_array]).unwrap();

        // Create partition field with identity transform on int
        let partition_field = PartitionField::new(1, 1000, "int_col", Transform::Identity);
        let struct_field = StructField {
            id: 1,
            name: "int_col".to_string(),
            required: true,
            field_type: Type::Primitive(PrimitiveType::Int),
            doc: None,
        };

        let bound_field = BoundPartitionField::new(&partition_field, &struct_field);

        // Partition the record batch
        let result = partition_record_batch(&record_batch, &[bound_field]).unwrap();
        let partitions: Vec<_> = result.collect::<Result<Vec<_>, _>>().unwrap();

        // Should have 2 partitions (100 and 200)
        assert_eq!(partitions.len(), 2);

        // Check partition values
        let partition_values: Vec<_> = partitions.iter().map(|(v, _)| v.clone()).collect();
        assert!(partition_values.contains(&vec![Value::Int(100)]));
        assert!(partition_values.contains(&vec![Value::Int(200)]));
    }

    #[test]
    fn test_identity_transform_date32() {
        // Create a record batch with Date32 column
        let schema = ArrowSchema::new(vec![
            Field::new("date_col", DataType::Date32, false),
            Field::new("value", DataType::Int32, false),
        ]);

        // Use dates: 2023-05-01 (19478), 2023-05-01 (19478), 2023-06-15 (19523)
        let date_array = Arc::new(Date32Array::from(vec![19478, 19478, 19523])) as ArrayRef;
        let value_array = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;

        let record_batch =
            RecordBatch::try_new(Arc::new(schema), vec![date_array, value_array]).unwrap();

        // Create partition field with identity transform on date
        let partition_field = PartitionField::new(1, 1000, "date_col", Transform::Identity);
        let struct_field = StructField {
            id: 1,
            name: "date_col".to_string(),
            required: true,
            field_type: Type::Primitive(PrimitiveType::Date),
            doc: None,
        };

        let bound_field = BoundPartitionField::new(&partition_field, &struct_field);

        // Partition the record batch
        let result = partition_record_batch(&record_batch, &[bound_field]).unwrap();
        let partitions: Vec<_> = result.collect::<Result<Vec<_>, _>>().unwrap();

        // Should have 2 partitions (19478 and 19523)
        assert_eq!(partitions.len(), 2);

        // Check partition values
        let partition_values: Vec<_> = partitions.iter().map(|(v, _)| v.clone()).collect();
        assert!(partition_values.contains(&vec![Value::Date(19478)]));
        assert!(partition_values.contains(&vec![Value::Date(19523)]));

        // Check record counts in each partition
        for (values, batch) in partitions {
            if values[0] == Value::Date(19478) {
                assert_eq!(batch.num_rows(), 2);
            } else if values[0] == Value::Date(19523) {
                assert_eq!(batch.num_rows(), 1);
            }
        }
    }
}
