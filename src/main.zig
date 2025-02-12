//! NOTE: my editor highlights "NOTE:" comments, hopefully yours does as well.
//! I tried to mark at least the more important changes in this format.
//!
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
// pub fn LayerMetadata() type {
// NOTE: No need for a function, types are values, having a function return a type is how zig implements generics but this is not generic.
pub const LayerMetadata = struct {
    allocator: std.mem.Allocator,
    name: []u8,
    shape: []u64,
    dtype: []u8,
    offset_start: u64,
    offset_end: u64,

    const Self = @This();

    // pub fn init(allocator: std.mem.Allocator, name: []const u8, shape: []u64, dtype: []const u8, offset_start: u64, offset_end: u64) !*Self {
    //     var self = try allocator.create(Self);
    //     self.allocator = allocator;
    //
    //     // Let's allocate memory for the struct fields so that
    //     // we do not depend on the json object and can free it after we exit this function scope.
    //     self.name = try allocator.alloc(u8, name.len);
    //     @memcpy(self.name, name);
    //     self.shape = try allocator.alloc(u64, shape.len);
    //     @memcpy(self.shape, shape);
    //     self.dtype = try allocator.alloc(u8, dtype.len);
    //     @memcpy(self.dtype, dtype);
    //     self.offset_start = offset_start;
    //     self.offset_end = offset_end;
    //     return self;
    // }
    //
    // pub fn deinit(self: *Self) void {
    //    self.allocator.free(self.name);
    //    self.allocator.free(self.shape);
    //    self.allocator.free(self.dtype);
    //    self.allocator.destroy(self);
    // }

    // NOTE: no need to allocate Self, defer this decision to the caller (otherwise you are forced to allocate to construct, which you might not want)
    // Also, allocator.dupe is convenient.
    pub fn init(allocator: std.mem.Allocator, name: []const u8, shape: []u64, dtype: []const u8, offset_start: u64, offset_end: u64) !Self {
        return Self{
            .name = try allocator.dupe(u8, name),
            .shape = try allocator.dupe(u64, shape),
            .dtype = try allocator.dupe(u8, dtype),
            .offset_start = offset_start,
            .offset_end = offset_end,
            .allocator = allocator,
        };
    }

    // NOTE: no need to pass by reference
    pub fn deinit(self: Self) void {
        self.allocator.free(self.name);
        self.allocator.free(self.shape);
        self.allocator.free(self.dtype);
    }

    // NOTE: no need to pass by reference
    pub fn print(self: LayerMetadata) void {
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
        const Self = @This();

        allocator: std.mem.Allocator,
        rows: usize,
        cols: usize,
        data: []T,

        // pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !*@This() {
        //     var self = try allocator.create(@This());
        //     self.rows = rows;
        //     self.cols = cols;
        //     self.data = try allocator.alloc(T, rows * cols);
        //     self.allocator = allocator;
        //     return self;
        // }
        // NOTE: return by value is fine here
        pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !Self {
            return Self{
                .rows = rows,
                .cols = cols,
                .data = try allocator.alloc(T, rows * cols),
                .allocator = allocator,
            };
        }

        // pub fn deinit(self: *@This()) void {
        pub fn deinit(self: Self) void {
            self.allocator.free(self.data);
            // self.allocator.destroy(self);
        }

        // pub fn at(self: *@This(), row: usize, col: usize) *T {
        // pub fn at(self: Self, row: usize, col: usize) *T {
        pub fn at(self: *const Self, row: usize, col: usize) *T {
            return &self.data[row * self.cols + col];
        }

        // pub fn copy_from(self: *@This(), array: *[]T) void {
        // NOTE: a few thoughts. You dont want to pass a pointer to a slice (which is itself a pointer)
        // Also, you calculate the index twice (once with `at()` and once inline).
        // Also, you have contiguous arrays with the same layout (row major) so you can just copy
        pub fn copy_from(self: Self, array: []T) void {
            // NOTE: a few ways of accomplishing this, here is just one (you could use the ptr capture you did layer on)
            std.mem.copyForwards(T, self.data, array);
            // for (0..self.rows) |row| {
            //     for (0..self.cols) |col| {
            //         // self.at(row, col).* = array.*[row * self.cols + col];
            //         self.at(row, col).* = array[row * self.cols + col];
            //     }
            // }
        }

        // pub fn print(self: *@This()) void {
        // NOTE: pass by value is fine here
        pub fn print(self: Self) void {
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

// NOTE: This creates `layers_info` so it doesnt have to be passed. However, one reason to pass `*std.ArrayList`
// might be that you want the caller to own `layers_info` so it can be reused
// (e.g. `layers_info.clearRetainingCapacity()`), but unclear if that was the idea.
pub fn get_safetensors_content(fpath: []const u8, allocator: std.mem.Allocator, layers_info: *std.ArrayList(LayerMetadata)) !u64 {
    // The code below is extremely hacky and was only tested on a model I mentioned above.
    // Ideally, we want a better JSON parser here, but it was not the goal.
    // Feel free to send PRs!
    var file = try std.fs.openFileAbsolute(fpath, .{});
    defer file.close();

    // Read 8 bytes first to get the header size.
    // We know the header size at comp time as it's the same for all safetensor files.
    var header_size_buf: [HEADER_SIZE_BUFF_SIZE]u8 = undefined;
    // _ = try file.read(header_size_buf[0..]);
    // NOTE: its more common to coerce an array to a slice like so...
    // see https://ziglang.org/documentation/master/#toc-Type-Coercion-Slices-Arrays-and-Pointers
    _ = try file.read(&header_size_buf); // recommend actually checking the number of bytes read

    // const header_size = std.mem.readInt(u64, &header_size_buf, std.builtin.Endian.little);
    // NOTE: Using enum shorthand notation is the convention (also useful if things get moved in new releases)
    const header_size = std.mem.readInt(u64, &header_size_buf, .little);
    std.debug.print("Header size: {d} bytes.\n", .{header_size});
    // Read the header.
    const header_buf = try allocator.alloc(u8, header_size);
    defer allocator.free(header_buf);

    // _ = try file.read(header_buf[0..]);
    // NOTE: pass slice directly, see above
    _ = try file.read(header_buf); // recommend actually checking the number of bytes read

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, header_buf, .{});
    defer parsed.deinit();
    var iter = parsed.value.object.iterator();

    while (iter.next()) |entry| {
        // NOTE: dereference immediately, no need to keep a ptr around and keep dereferencing it (also, constness)
        const key = entry.key_ptr.*;
        // Skip metadata, we need only layers.
        if (std.mem.eql(u8, key, "__metadata__")) {
            continue;
        }
        const val = entry.value_ptr.*; // NOTE: same as above
        const dtype = val.object.get("dtype").?.string;

        const unk_shape = val.object.get("shape").?.array;
        // NOTE: this pattern of allocating and freeing in a loop is a bit of a code smell in zig (but logically fine)
        const shape = try allocator.alloc(u64, unk_shape.items.len);
        defer allocator.free(shape);
        // Shape is an array of integers, as we do not know the amount of items,
        // we need to allocate it explicitly on the heap.
        // We can probably have a max number and store the shape dims as well, but I took this path.
        for (unk_shape.items, 0..) |el, idx| {
            switch (el) {
                .integer => |num| {
                    // const signless: u64 = @as(u64, @intCast(num));
                    // shape[idx] = @intCast(signless);
                    // NOTE: zig can infer type and perform the cast
                    shape[idx] = @intCast(num);
                },
                else => {},
            }
        }

        // With offsets, we actually know that there is only start and end.
        // Let's exploit this and store those as separate fields in the struct.
        // const offset_start: u64 = -1;
        // const offset_end: u64 = -1;
        const unk_offsets = val.object.get("data_offsets").?.array;
        // NOTE: you only one of these u64's to inform the compiler what cast you want
        // const offset_start: u64 = @as(u64, @intCast(unk_offsets.items[0].integer));
        // const offset_end: u64 = @as(u64, @intCast(unk_offsets.items[1].integer));
        const offset_start: u64 = @intCast(unk_offsets.items[0].integer);
        const offset_end: u64 = @intCast(unk_offsets.items[1].integer);
        const cur_layer = try LayerMetadata.init(
            allocator,
            // key.*[0..key.*.len],
            key, // NOTE: no need to slice the slice to pass a slice :)
            shape,
            dtype,
            offset_start,
            offset_end,
        );
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

/// Get weights for a particular layer.
// NOTE: pass layer_metadata by value
pub fn load_weights(header_size: u64, layer_metadata: LayerMetadata, safetensors_path: []const u8, allocator: std.mem.Allocator) !NDArray(f32) {
    // Let's now take one layer and print it out.
    // We will need to read bytes from the file using the offset info
    // in the LayerMetadata struct.
    var file = try std.fs.openFileAbsolute(safetensors_path, .{});
    defer file.close();

    const metadata_bytesize = HEADER_SIZE_BUFF_SIZE + header_size;
    const read_len = layer_metadata.offset_end - layer_metadata.offset_start;

    // Weight offsets are starting with 0 meaning the first byte after the header.
    try file.seekTo(layer_metadata.offset_start + metadata_bytesize);
    const wbuf = try allocator.alloc(u8, read_len);
    const bytes_read = try file.read(wbuf);
    if (bytes_read != read_len) { // NOTE: nice!
        // NOTE: this is not how you want to error in zig. If it's unrecoverable then use panic, otherwise use an error type
        // std.debug.print("Something is wrong! Expected bytes to read: {}. Actual bytes read:{}.]\n", .{ read_len, bytes_read });
        // std.process.exit(1);
        std.debug.panic("Something is wrong! Expected bytes to read: {}. Actual bytes read:{}.]\n", .{ read_len, bytes_read });
    }
    defer allocator.free(wbuf); // NOTE: move this line up, otherwise you could leak (e.g. if file.read errors)

    const bf16_count: usize = read_len / 2;
    const rows = layer_metadata.shape[0];
    const cols = layer_metadata.shape[1];
    // Again, here we assume that we have a two dimensional array.
    // Check that the shape corresponds to amount of bytes we read.
    std.debug.assert(rows * cols == bf16_count);

    // Original weights are in bf16, let's get fp32 from those.
    // const f32_values = try allocator.alloc(f32, bf16_count); // NOTE: can now be const
    // defer allocator.free(f32_values);
    // NOTE: if we construct weights ourselves (described below) then dont deinit
    const f32_values = try allocator.alloc(f32, bf16_count); // NOTE: can now be const
    batch_bf16bytes_to_fp32(wbuf, bf16_count, f32_values);

    // Let's get the 2D array printed to compare to what we see in Python (run test.py to compare).
    // NOTE: could do this (but another option below)
    // const weights = try allocator.create(NDArray(f32));
    // weights.* = try NDArray(f32).init(allocator, rows, cols);
    //
    // var weights = try NDArray(f32).init(allocator, rows, cols);
    // weights.copy_from(&f32_values);
    // weights.copy_from(f32_values); // NOTE: slice is already a pointer
    //
    // NOTE: one thing to consider might be creating an NDArray without using init().
    // This allows you to avoid extra allocations (can be easier to reason about).
    // f32_values are already allocated so we can create the struct directly
    return NDArray(f32){
        .allocator = allocator,
        .rows = rows,
        .cols = cols,
        .data = f32_values,
    };
}

/// This function returns the weights values given the safetensors file path and a layer name.
/// I need this dependency in my other code, this is to be used as external library.
pub fn extract_weights(safetensors_path: []const u8, layer_name: []const u8, allocator: std.mem.Allocator) !NDArray(f32) {
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

    // At this point, we will know all the layers names, their types, shapes, and offsets.
    // NOTE: No need to store values as pointers here, it makes ownership and lifetimes harder to reason about
    // and will have worse performance.
    var layers_info = std.ArrayList(LayerMetadata).init(allocator);
    defer {
        // NOTE: using the pointer capture is good for when you need a pointer, but you just needed the value
        // and immediately deferenced it, so we can just capture that value normally.
        // for (layers_info.items) |*item| {
        //     item.*.deinit();
        for (layers_info.items) |item| item.deinit();
        layers_info.deinit();
    }
    // NOTE: as mentioned above, not so sure I would create layers_info this way but it depends what you had in mind
    const header_size = try get_safetensors_content(safetensors_path, allocator, &layers_info);

    // NOTE: since we changed LayerMetadata's init to not allocate and return a ptr,
    // we have some options to refactor this. Using a block is clean.
    // Also, `layer_metadata` could easily be left `undefined` here which would be a huge problem.
    // It is often said in regards to undefined behavior that "a crash is the best possible outcome"
    // var layer_metadata: *LayerMetadata() = undefined;
    // for (layers_info.items) |layer_spec| {
    //     layer_spec.print();
    //     if (std.mem.eql(u8, layer_spec.name, layer_name)) {
    //         layer_metadata = layer_spec;
    //     }
    // }
    for (layers_info.items) |layer_spec| layer_spec.print();
    var layer_metadata = blk: {
        for (layers_info.items) |layer_spec| if (std.mem.eql(u8, layer_spec.name, layer_name)) break :blk layer_spec;
        // Do something...
        std.debug.panic("yikes", .{});
    };
    layer_metadata.print();
    // NOTE: `load_weights` can error! use try or handle it here
    // We can extract the weights now.
    // return load_weights(header_size, layer_metadata, safetensors_path, allocator);
    return try load_weights(header_size, layer_metadata, safetensors_path, allocator);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // NOTE: it's a good idea to deinit your gpa and check the return value
    defer switch (gpa.deinit()) {
        .leak => std.debug.panic("Leaked", .{}),
        .ok => {},
    };
    const allocator = gpa.allocator();
    var weights = try extract_weights(SAFETENSORS_FPATH, LAYER_TO_CONVERT, allocator);
    weights.print();
    weights.deinit();
}

// NOTE: in a test block0xxc
test {
    const allocator = std.testing.allocator;
    var weights = try extract_weights(SAFETENSORS_FPATH, LAYER_TO_CONVERT, allocator);
    weights.print();
    weights.deinit();
}
