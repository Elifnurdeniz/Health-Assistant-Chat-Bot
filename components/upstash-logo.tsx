/* eslint-disable @typescript-eslint/no-unused-vars */

import React, { HTMLProps } from "react";

export interface Props extends HTMLProps<SVGSVGElement> {
  size?: number;
}

export default function UpstashLogo({ height = 20, ...props }: Props) {
  return (
    <svg
      role="img"
      height={height}
      viewBox="0 0 118 118"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <g clipPath="url(#upstash_icon_dark_bg)">
        <path
          className="fill-emerald-400"
          d="M15.105 103.244c19.416 19.526 50.895 19.526 70.311 0 19.416-19.526 19.416-51.185 0-70.711l-8.789 8.839c14.562 14.645 14.562 38.388 0 53.033-14.562 14.644-38.171 14.644-52.733 0l-8.789 8.839Z"
        />
        <path
          className="fill-emerald-400"
          d="M32.683 85.566c9.708 9.763 25.447 9.763 35.155 0 9.708-9.763 9.708-25.592 0-35.355L59.05 59.05c4.854 4.881 4.854 12.796 0 17.677a12.38 12.38 0 0 1-17.578 0l-8.79 8.839Z"
        />
        <path
          className="fill-emerald-200"
          d="M102.994 14.855c-19.416-19.526-50.895-19.526-70.311 0-19.416 19.527-19.416 51.185 0 70.711l8.788-8.839c-14.561-14.645-14.561-38.388 0-53.033 14.562-14.644 38.172-14.644 52.734 0l8.789-8.839Z"
        />
        <path
          className="fill-emerald-200"
          d="M85.416 32.533c-9.708-9.763-25.448-9.763-35.156 0-9.708 9.763-9.708 25.592 0 35.355l8.79-8.839c-4.855-4.881-4.855-12.795 0-17.677a12.38 12.38 0 0 1 17.577 0l8.789-8.839Z"
        />
      </g>
      <defs>
        <clipPath id="upstash_icon_dark_bg">
          <path fill="#fff" d="M15 0h88v118H15z" />
        </clipPath>
      </defs>
    </svg>
  );
}