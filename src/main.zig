//! This code allows you to load safetensors model natively in Zig.
//! Some of the stuff is hardcoded intentionally to make the code simpler.
//! I also avoided any external dependencies (ndarrays, arg parsing etc)
//! for the same reasons.
//! This is also my first Zig project after 2 days of doing Ziglings.
//! So, if you have any ideas to improve this, submit a PR or send me an email.
//! You can find me at y0b1byte@proton.me

const std = @import("std");

const HEADER_SIZE_BUFF_SIZE = 8;
// I am using this model:
// https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/blob/main/model.safetensors
const SAFETENSORS_FPATH = "/home/yobibyte/Downloads/model.safetensors";
const LAYER_TO_CONVERT = "model.layers.9.self_attn.v_proj.weight";

// Safetensors header has a JSON with metadata about our layers.
// They all have a name, a shape, dtype, and offsets which tell us
// how to find the bytes for those weights in the binary file.
const LayerMetadata = struct {
    name: []const u8,
    shape: []u64,
    dtype: []const u8,
    offset_start: u64,
    offset_end: u64,

    pub fn print(self: *const LayerMetadata) void {
        std.debug.print("{s}\n  dtype:{s}\n", .{ self.name, self.dtype });

        std.debug.print("  shape: ", .{});
        for (self.shape) |el| {
            std.debug.print("{d} ", .{el});
        }
        std.debug.print("\n", .{});
        std.debug.print("  data_offsets: ", .{});
        std.debug.print("{d} {d} ", .{ self.offset_start, self.offset_end });

        std.debug.print("\n\n", .{});
    }
};

/// A struct that makes you belive that it is a two dimensional array.
/// In practice, it's a 1D array with methods to access it as 2D.
/// This uses row-based layout in memory, i.e. the element at i,j is
/// array[i*columns + j].
/// Also, there is no bounds check here, it is as unsafe as it can be.
/// DID YOU THINK IT WAS RUST??????
/// We are living on the edge.
pub fn NDArray(comptime T: type) type {
    return struct {
        allocator: *const std.mem.Allocator,
        rows: usize,
        cols: usize,
        data: []T,

        pub fn init(allocator: *std.mem.Allocator, rows: usize, cols: usize) !*@This() {
            var self = try allocator.create(@This());
            self.rows = rows;
            self.cols = cols;
            self.data = try allocator.alloc(T, rows * cols);
            self.allocator = allocator;
            return self;
        }

        pub fn deinit(self: *@This()) void {
            self.allocator.free(self.data);
        }

        pub fn at(self: *@This(), row: usize, col: usize) *T {
            return &self.data[row * self.cols + col];
        }

        pub fn copy_from(self: *@This(), array: *[]T) void {
            for (0..self.rows) |row| {
                for (0..self.cols) |col| {
                    self.at(row, col).* = array.*[row * self.cols + col];
                }
            }
        }

        pub fn print(self: *@This()) void {
            //check that row, col > 3 or print all
            for (0..self.rows) |row| {
                if (self.rows < 3 or row < 3 or row > self.rows - 4) {
                    for (0..self.cols) |col| {
                        if (self.cols < 3 or col < 3 or col > self.cols - 4) {
                            std.debug.print("{d:.4} ", .{self.at(row, col).*});
                            if (col == self.cols - 1) {
                                std.debug.print("\n", .{});
                            }
                        }
                        if (col == 4 or col == 5) {
                            std.debug.print("..., ", .{});
                        }
                    }
                }
                if (row == 4 or row == 5) {
                    std.debug.print("..., \n", .{});
                }
            }
        }
    };
}

pub fn get_safetensors_content(fpath: []const u8, allocator: *std.mem.Allocator, layers_info: *std.ArrayList(LayerMetadata)) !u64 {
    // The code below is extremely hacky and was only tested on a model I mentioned above.
    // Ideally, we want a better JSON parser here, but it was not the goal.
    // Feel free to send PRs!
    var file = try std.fs.openFileAbsolute(fpath, .{});
    defer file.close();

    // Read 8 bytes first to get the header size.
    // We know the header size at comp time as it's the same for all safetensor files.
    var header_size_buf: [HEADER_SIZE_BUFF_SIZE]u8 = undefined;
    _ = try file.read(header_size_buf[0..]);

    const header_size = std.mem.readInt(u64, &header_size_buf, std.builtin.Endian.little);
    std.debug.print("Header size: {d} bytes.\n", .{header_size});
    // Read the header.
    const header_buf = try allocator.alloc(u8, header_size);
    defer allocator.free(header_buf);
    _ = try file.read(header_buf[0..]);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator.*, header_buf, .{});
    defer parsed.deinit();
    var iter = parsed.value.object.iterator();

    while (iter.next()) |entry| {
        const key = entry.key_ptr;
        // Skip metadata, we need only layers.
        if (std.mem.eql(u8, key.*, "__metadata__")) {
            continue;
        }
        const val = entry.value_ptr;
        const dtype = val.object.get("dtype").?.string;

        const unk_shape = val.object.get("shape").?.array;
        const shape = try allocator.alloc(u64, unk_shape.items.len);
        // Shape is an array of integers, as we do not know the amount of items,
        // we need to allocate it explicitly on the heap.
        // We can probably have a max number and store the shape dims as well, but I took this path.
        for (unk_shape.items, 0..) |el, idx| {
            switch (el) {
                .integer => |num| {
                    const signless: u64 = @as(u64, @intCast(num));
                    shape[idx] = @intCast(signless);
                },
                else => {},
            }
        }

        // With offsets, we actually know that there is only start and end.
        // Let's exploit this and store those as separate fields in the struct.
        // const offset_start: u64 = -1;
        // const offset_end: u64 = -1;
        const unk_offsets = val.object.get("data_offsets").?.array;
        const offset_start: u64 = @as(u64, @intCast(unk_offsets.items[0].integer));
        const offset_end: u64 = @as(u64, @intCast(unk_offsets.items[1].integer));

        // Let's allocate memory for the struct fields so that
        // we do not depend on the json object and can free it after we exit this function scope.
        const name = try allocator.alloc(u8, key.*.len);
        @memcpy(name, key.*[0..key.*.len]);
        const dtype_cpy = try allocator.alloc(u8, dtype.len);
        @memcpy(dtype_cpy, dtype);
        const shape_cpy = try allocator.alloc(u64, shape.len);
        @memcpy(shape_cpy, shape);

        const cur_layer = LayerMetadata{
            .name = name,
            .dtype = dtype_cpy,
            .offset_start = offset_start,
            .offset_end = offset_end,
            .shape = shape_cpy,
        };
        try layers_info.append(cur_layer);
    }
    return header_size;
}

/// Get a slice of bytes representing bf16 and convert them to a slice of fp32.
/// bf16 and fp32 have the same sign+exponent -> we can just treat f32 as a bf16 padded with zeros from the right side.
pub fn batch_bf16bytes_to_fp32(bf16_buf: []u8, bf16_count: usize, fp32_buf: []f32) void {
    const bf16_ptr = @as([*]u16, @ptrCast(@alignCast(bf16_buf.ptr)));
    const bf16_slice = bf16_ptr[0..bf16_count];
    const shift_width: u32 = 16;
    for (bf16_slice, 0..) |bf, i| {
        const bits: u32 = @as(u32, bf) << shift_width;
        fp32_buf[i] = @bitCast(bits);
    }
}

pub fn main() !void {
    // https://huggingface.co/docs/safetensors/index <- Useful info on safetensors.
    // Safetensors TLDR: | HEADER SIZE (N)   | HEADER JSON | NUMBERS
    //                     ^^^ 8 bytes (u64)      N bytes  ^
    //                                                     |
    //                                                      ____ Offset 0 is here.
    // The idea:
    // 1. Get the header size.
    // 2. Read the header size, convert to UTF-8, parse JSON.
    // 3. Header items have everything we need: model name, dtype, shapes, and offsets.
    // Offset start tells us where to start reading in the file to get the numbers.
    // dtype tells use how much bytes we need to read and what to cast the numbers to.

    // Read the header and parse the JSON.
    var allocator = std.heap.page_allocator;

    var layers_info = std.ArrayList(LayerMetadata).init(allocator);
    defer {
        for (layers_info.items) |*item| {
            allocator.free(item.shape);
        }
        layers_info.deinit();
    }

    var file = try std.fs.openFileAbsolute(SAFETENSORS_FPATH, .{});
    defer file.close();

    const header_size = try get_safetensors_content(SAFETENSORS_FPATH, &allocator, &layers_info);

    for (layers_info.items) |layer_spec| {
        layer_spec.print();
    }

    // Let's now take one layer and print it out.
    // We will need to read bytes from the file using the offset info
    // in the LayerMetadata struct.
    var layer_metadata: LayerMetadata = undefined;
    for (layers_info.items) |layer_spec| {
        if (std.mem.eql(u8, layer_spec.name, LAYER_TO_CONVERT)) {
            layer_metadata = layer_spec;
        }
    }
    layer_metadata.print();

    const metadata_bytesize = HEADER_SIZE_BUFF_SIZE + header_size;
    const read_len = layer_metadata.offset_end - layer_metadata.offset_start;

    // Weight offsets are starting with 0 meaning the first byte after the header.
    try file.seekTo(layer_metadata.offset_start + metadata_bytesize);
    const wbuf = try allocator.alloc(u8, read_len);
    const bytes_read = try file.read(wbuf);
    if (bytes_read != read_len) {
        std.debug.print("Something is wrong! Expected bytes to read: {}. Actual bytes read:{}.]\n", .{ read_len, bytes_read });
        std.process.exit(1);
    }
    defer allocator.free(wbuf);

    const bf16_count: usize = read_len / 2;
    const rows = layer_metadata.shape[0];
    const cols = layer_metadata.shape[1];
    // Again, here we assume that we have a two dimensional array.
    // Check that the shape corresponds to amount of bytes we read.
    std.debug.assert(rows * cols == bf16_count);

    // Original weights are in bf16, let's get fp32 from those.
    var f32_values = try allocator.alloc(f32, bf16_count);
    defer allocator.free(f32_values);
    batch_bf16bytes_to_fp32(wbuf, bf16_count, f32_values);

    // Let's get the 2D array printed to compare to what we see in Python (run test.py to compare).
    var arr_allocator = std.heap.page_allocator;
    var weights = try NDArray(f32).init(&arr_allocator, rows, cols);
    defer weights.deinit();
    weights.copy_from(&f32_values);
    weights.print();
}
