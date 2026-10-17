//! Renders chat cases with TensorFold's native template engine (zig/src/core/template), for check.py.
//! render_native TEMPLATE CASES.json OUT.json: CASES is [{messages, tools, context, add_generation_prompt}], OUT the
//! rendered strings (or {"error": message}) in order.
const std = @import("std");
const template = @import("template");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) return error.ExpectedTemplateCasesOut;
    const src = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], gpa, .limited(1 << 20));
    defer gpa.free(src);
    const raw = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], gpa, .limited(64 << 20));
    defer gpa.free(raw);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = try std.json.parseFromSliceLeaky(std.json.Value, a, raw, .{});
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, "[");
    for (cases.array.items, 0..) |case, i| {
        if (i > 0) try out.appendSlice(a, ",");
        const o = case.object;
        var diag: template.Diag = .{};
        const gen = if (o.get("add_generation_prompt")) |g| g.bool else true;
        const text = template.renderChat(gpa, src, o.get("messages").?, o.get("tools") orelse .null, o.get("context") orelse .null, gen, &diag) catch {
            try out.print(a, "{f}", .{std.json.fmt(.{ .@"error" = diag.msg }, .{})});
            continue;
        };
        defer gpa.free(text);
        try out.print(a, "{f}", .{std.json.fmt(text, .{})});
    }
    try out.appendSlice(a, "]");
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[3], .data = out.items });
}
